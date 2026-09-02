"""Run integrated WarpSDF + TacMap tactile visualization.

Example:
    CONDA_PREFIX=/home/abc/miniforge3/envs/env_isaaclab TERM=xterm \
    /home/abc/IsaacLab/isaaclab.sh -p integrate/run_integrated_tactile.py \
      --mode press --finger middle --press-start-offset 0.025 \
      --press-end-offset 0.018 --press-steps 700
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


INTEGRATE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATE_ROOT.parent
OFFICIAL_REPLAY_ROOT = REPO_ROOT / "scripts" / "force_map" / "official_replay"
TACMAP_ROOT = REPO_ROOT / "tacmap"
DEFAULT_TACEX_RGB_ROOT = INTEGRATE_ROOT / "third_party" / "tacex_gpu_taxim"
DEFAULT_TACEX_RGB_CALIB_DIR = DEFAULT_TACEX_RGB_ROOT / "calibs" / "640x480"
DEFAULT_TACEX_RGB_SIM_DIR = DEFAULT_TACEX_RGB_ROOT / "sim"
DEFAULT_PRESSURE_PAD_LAYOUT_URDF = (
    REPO_ROOT / "assets" / "revo21_right_touch" / "urdf" / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
)
DEFAULT_DENSE_REFERENCE_ACCEPTANCE_LAYERS = ("L0_analytic", "L1_model_reference", "L2_offline_oracle")
FINGER_CHOICES = ("middle", "index", "ring", "pinky", "thumb")
PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
FOCUS_PRESSURE_PAD_SEGMENTS = ("mcp", "pip")

for path in (INTEGRATE_ROOT, OFFICIAL_REPLAY_ROOT, TACMAP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tensor_image_utils import (  # noqa: E402
    combined_image as _combined_image_t,
    depth_weighted_center_px as _depth_weighted_center_px_t,
    fixed_scale_grid as _fixed_scale_grid_t,
    force_strip as _force_strip_t,
    image_strip as _image_strip_t,
    jet_colormap as _jet_colormap_t,
    normalize_grid as _normalize_grid_t,
    resize_nearest as _resize_nearest_t,
    resize_nearest_to_shape as _resize_nearest_to_shape_t,
    tacmap_strip as _tacmap_strip_t,
    to_numpy_float32 as _tensor_to_numpy_float32,
    to_numpy_uint8_rgb,
)

from pressure_calibration_setup import (  # noqa: E402
    BALL_PROBE_PRESSURE_PAD_QUAT_WXYZ,
    normalize_signed_axis_args,
    press_depth_stats_line,
    print_press_coordinates,
    print_rubber_link_poses,
)


sys.argv = [sys.argv[0], *normalize_signed_axis_args(sys.argv[1:])]


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("points", "press"), default="press")
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--finger", choices=FINGER_CHOICES, default="middle")
parser.add_argument(
    "--fingers",
    type=str,
    default=None,
    help=(
        "Comma-separated Revo21 DIP rubber tactile links to enable, e.g. "
        "middle,index,ring,pinky,thumb. The first finger is used as the scripted press-motion reference "
        "unless --focus-finger is set."
    ),
)
parser.add_argument(
    "--focus-finger",
    choices=FINGER_CHOICES,
    default=None,
    help="Finger used for both scripted presser targeting and integrated pane display.",
)
parser.add_argument("--robot-urdf", type=str, default=None, help="Optional robot hand URDF path.")
parser.add_argument(
    "--robot-usd-output-dir",
    type=str,
    default=None,
    help="Optional isolated USD conversion output directory for --robot-urdf.",
)
parser.add_argument("--press-start-offset", type=float, default=0.025)
parser.add_argument("--press-end-offset", type=float, default=0.018)
parser.add_argument(
    "--press-distance",
    type=float,
    default=None,
    help="Calibration shorthand in meters: set press_end_offset = press_start_offset - distance.",
)
parser.add_argument(
    "--press-indent-depth",
    type=float,
    default=None,
    help="After first WarpSDF contact onset, press this many meters deeper.",
)
parser.add_argument(
    "--press-contact-search-distance",
    type=float,
    default=None,
    help="Maximum pre-contact approach travel in meters for --press-indent-depth.",
)
parser.add_argument(
    "--press-contact-threshold",
    type=float,
    default=1.0e-7,
    help="WarpSDF penetration threshold in meters used as contact onset for --press-indent-depth.",
)
parser.add_argument("--press-steps", type=int, default=240)
parser.add_argument("--press-slide-distance", type=float, default=0.0)
parser.add_argument("--press-slide-steps", type=int, default=0)
parser.add_argument("--press-motion-frame", choices=("link_surface", "touch"), default=None)
parser.add_argument(
    "--press-slide-axis",
    choices=("+u", "-u", "+v", "-v", "+ray", "-ray", "+x", "-x", "+y", "-y", "+z", "-z"),
    default="+u",
)
parser.add_argument(
    "--press-motion-source",
    choices=("pressure_layout", "legacy_finger_map"),
    default=None,
    help=(
        "Source for the scripted presser center/normal. Defaults to legacy_finger_map for the original "
        "integrated TacMap/FOTS demos, and pressure_layout when --pressure-layout-urdf is provided."
    ),
)
parser.add_argument("--press-center-l", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--press-center-offset-l", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"))
parser.add_argument("--press-normal-l", type=float, nargs=3, default=None, metavar=("NX", "NY", "NZ"))
parser.add_argument("--robot-world-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--robot-world-quat-wxyz",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
)
parser.add_argument(
    "--setup-edit-robot-pose",
    action="store_true",
    help="During --press-setup-before-motion, let GUI edits to /World/Robot set the hand root pose.",
)
parser.add_argument(
    "--print-rubber-link-poses",
    action="store_true",
    help="Print world poses for robot rubber/tubber links after reset.",
)
parser.add_argument("--live-control-file", type=str, default=None)
parser.add_argument("--write-live-control-template", action="store_true")
parser.add_argument("--lock-press-finger-joints", action="store_true")
parser.add_argument("--presser", choices=("cylinder_D4", "square_4", "ball_probe"), default="cylinder_D4")
parser.add_argument("--presser-scale", type=float, default=1.0)
parser.add_argument("--presser-contact-offset", type=float, default=0.000001)
parser.add_argument("--presser-rest-offset", type=float, default=-0.00075)
parser.add_argument("--presser-mass", type=float, default=0.2)
presser_collision_group = parser.add_mutually_exclusive_group()
presser_collision_group.add_argument(
    "--enable-presser-collision",
    dest="presser_collision_enabled",
    action="store_true",
    help="Let /World/Plug participate in PhysX contact. Needed for PhysX contact-map diagnostics.",
)
presser_collision_group.add_argument(
    "--disable-presser-collision",
    dest="presser_collision_enabled",
    action="store_false",
    help="Keep /World/Plug visible/queryable for SDF, but remove it from PhysX collision solving.",
)
parser.set_defaults(presser_collision_enabled=None)
plug_fix_group = parser.add_mutually_exclusive_group()
plug_fix_group.add_argument(
    "--plug-fix-base",
    dest="plug_fix_base",
    action="store_true",
    help="Spawn /World/Plug as fixed-base so GUI dragging is not fought by gravity/dynamics.",
)
plug_fix_group.add_argument(
    "--plug-free-base",
    dest="plug_fix_base",
    action="store_false",
    help="Spawn /World/Plug as a dynamic free-base object.",
)
parser.set_defaults(plug_fix_base=None)
parser.add_argument(
    "--press-object-control",
    choices=("scripted", "initial_only", "manual_gui"),
    default="scripted",
    help=(
        "How /World/Plug is driven in press mode. scripted writes the scripted pose every frame; "
        "initial_only writes the scripted initial pose once and then lets the GUI/user move it; "
        "manual_gui never writes scripted poses and uses the world initial pose."
    ),
)
parser.add_argument(
    "--press-motion-actor",
    choices=("object", "finger", "hand"),
    default="object",
    help=(
        "Which side performs the scripted press. object keeps the finger fixed and moves /World/Plug; "
        "finger keeps /World/Plug fixed after setup and moves the selected robot joint(s); "
        "hand keeps /World/Plug fixed and moves the whole hand root along the selected sensor normal."
    ),
)
parser.add_argument(
    "--press-hand-axis-link",
    type=str,
    default=None,
    help=(
        "Move along this robot link's local axis; use 'world' for a world-frame axis. "
        "Pressure-pad ball_probe object press defaults to the probe STL local -Y axis."
    ),
)
parser.add_argument(
    "--press-hand-axis",
    choices=("+x", "-x", "+y", "-y", "+z", "-z"),
    default="+x",
    help="Local axis used with --press-hand-axis-link. If no link is set, the pressure pad normal is used.",
)
parser.add_argument(
    "--press-finger-joint",
    action="append",
    default=None,
    help=(
        "Robot joint to drive when --press-motion-actor finger is used. "
        "May be repeated. Defaults to right_midmcp_roll_joint."
    ),
)
parser.add_argument(
    "--press-finger-start-rad",
    type=float,
    default=0.0,
    help="Start angle for each --press-finger-joint in --press-motion-actor finger mode.",
)
parser.add_argument(
    "--press-finger-end-rad",
    type=float,
    default=0.2,
    help="End angle for each --press-finger-joint in --press-motion-actor finger mode.",
)
parser.add_argument(
    "--disable-press-finger-kinematic-sensor-pose",
    action="store_true",
    help=(
        "In finger actor mode, do not rewrite the commanded finger pose immediately before WarpSDF sensor update. "
        "Useful only for comparing against post-PhysX-solver rebound."
    ),
)
parser.add_argument(
    "--press-hold-joint-pose",
    choices=("zero", "default"),
    default="zero",
    help="Joint pose used when a press-only alternate robot URDF does not match the replay dataset joint names.",
)
parser.add_argument(
    "--press-initial-joint-deg",
    action="append",
    nargs=2,
    default=None,
    metavar=("JOINT", "DEG"),
    help="Override one held press-mode joint angle in degrees. May be repeated.",
)
parser.add_argument(
    "--press-setup-before-motion",
    action="store_true",
    help="Open an editable GUI setup phase before starting the scripted press trajectory.",
)
parser.add_argument(
    "--press-setup-start-file",
    type=str,
    default=None,
    help="Optional file trigger for --press-setup-before-motion; create the file to start motion.",
)
parser.add_argument(
    "--press-debug-log",
    type=str,
    default=None,
    help="Write structured JSONL press/debug telemetry for GUI diagnosis.",
)
parser.add_argument(
    "--press-debug-log-every",
    type=int,
    default=5,
    help="Frame interval for --press-debug-log. Event records are always written.",
)
parser.add_argument("--presser-world-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--presser-world-quat-wxyz",
    type=float,
    nargs=4,
    default=None,
    metavar=("W", "X", "Y", "Z"),
)
parser.add_argument(
    "--presser-pose-save-file",
    type=str,
    default=None,
    help="Write the current /World/Plug world pose to JSON while running, useful after GUI dragging.",
)
parser.add_argument(
    "--presser-pose-save-every",
    type=int,
    default=5,
    help="Frame interval for --presser-pose-save-file.",
)
parser.add_argument("--flip-axis", choices=("auto", "none", "x", "y", "z"), default="auto")
parser.add_argument("--presser-extra-rot-axis", choices=("auto", "none", "x", "y", "z"), default="auto")
parser.add_argument("--presser-extra-rot-deg", type=float, default=None)
parser.add_argument("--disable-touch-compliant-material", action="store_true")
parser.add_argument("--touch-stiffness", type=float, default=280.0)
parser.add_argument("--touch-damping", type=float, default=20.0)
parser.add_argument("--touch-contact-offset", type=float, default=0.0002)
parser.add_argument("--touch-rest-offset", type=float, default=-0.0001)
parser.add_argument("--show-sample-points", action="store_true")
parser.add_argument("--show-sample-axes", action="store_true")
parser.add_argument("--sample-point-radius", type=float, default=0.0015)
parser.add_argument("--show-pressure-pad-centers", action="store_true")
parser.add_argument("--pressure-pad-center-radius", type=float, default=0.003)
parser.add_argument(
    "--show-pressure-pad-taxel-points",
    action="store_true",
    help="Show every URDF-declared pressure-pad taxel point as a yellow marker.",
)
parser.add_argument(
    "--hide-pressure-pad-taxel-points",
    action="store_true",
    help="Disable the default yellow taxel-point markers in legacy integrated runs.",
)
parser.add_argument("--pressure-pad-taxel-point-radius", type=float, default=0.00045)
parser.add_argument("--print-press-coordinates", action="store_true")
parser.add_argument("--press-coordinate-log-every", type=int, default=30)
parser.add_argument("--num-rows", type=int, default=None)
parser.add_argument("--num-cols", type=int, default=None)
parser.add_argument("--point-distance", type=float, default=None)
parser.add_argument("--row-distance", type=float, default=None)
parser.add_argument("--col-distance", type=float, default=None)
parser.add_argument("--normal-axis", type=int, choices=(0, 1, 2), default=None)
parser.add_argument("--normal-offset", type=float, default=None)
parser.add_argument("--normal-sign", type=float, default=None)
parser.add_argument("--disable-tacmap", action="store_true")
parser.add_argument("--tacmap-resolution-step", type=int, default=1)
parser.add_argument("--tacmap-max-distance", type=float, default=0.008)
parser.add_argument("--tacmap-invert-normals", action="store_true")
parser.add_argument("--tacmap-use-original-normals", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--tacmap-debug-hit-stats", action="store_true")
parser.add_argument("--tacmap-debug-hit-every", type=int, default=50)
parser.add_argument("--show-tacmap-points", action="store_true")
parser.add_argument("--show-tacmap-active-points", action="store_true")
parser.add_argument("--show-tacmap-rays", action="store_true")
parser.add_argument("--show-tacmap-link-surfaces", action="store_true")
parser.add_argument("--tacmap-active-point-threshold", type=float, default=0.0)
parser.add_argument("--no-local-ui", action="store_true")
parser.add_argument("--force-scale", type=int, default=8)
parser.add_argument(
    "--focus-pressure-pad-tile-size",
    type=int,
    default=240,
    help="Display size in pixels for each focus-finger pressure-pad pane; <=0 keeps --force-scale only.",
)
parser.add_argument(
    "--focus-pressure-pad-presser-segment",
    choices=FOCUS_PRESSURE_PAD_SEGMENTS,
    default="mcp",
    help="Which focus-finger pressure pad receives the extra pressure-pad presser.",
)
parser.add_argument(
    "--enable-focus-pressure-pad-presser",
    action="store_true",
    help="Spawn an extra presser object for one pressure pad on --focus-finger.",
)
parser.add_argument("--disable-focus-pressure-pad-presser", action="store_true")
parser.add_argument("--pressure-pad-press-start-offset", type=float, default=None)
parser.add_argument("--pressure-pad-press-end-offset", type=float, default=None)
parser.add_argument("--pressure-pad-press-steps", type=int, default=None)
parser.add_argument("--focus-pressure-pad-presser-start-offset", type=float, default=0.012)
parser.add_argument("--focus-pressure-pad-presser-end-offset", type=float, default=-0.001)
parser.add_argument(
    "--focus-pressure-pad-presser-offset-l",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Extra pressure-pad presser position offset in the target pressure-pad link frame, meters.",
)
parser.add_argument(
    "--focus-pressure-pad-presser-start-offset-l",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Explicit pressure-pad presser start offset from the target pressure-pad center in link frame, meters.",
)
parser.add_argument(
    "--focus-pressure-pad-presser-end-offset-l",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Explicit pressure-pad presser end offset from the target pressure-pad center in link frame, meters.",
)
parser.add_argument("--focus-pressure-pad-presser-scale", type=float, default=None)
parser.add_argument("--tacmap-scale", type=int, default=1)
parser.add_argument("--force-gamma", type=float, default=0.7)
parser.add_argument("--pressure-view-source", choices=("warpsdf", "normal_ray", "geometry_normal_ray"), default="warpsdf")
parser.add_argument(
    "--pressure-contact-model",
    choices=("signed_penetration", "surface_gap"),
    default=None,
    help=(
        "Pressure backend contact model. signed_penetration uses negative signed SDF depth; "
        "surface_gap uses unsigned mesh gap inside --mesh-shell-thickness, so pressure can "
        "activate without requiring visible rigid-body interpenetration. Defaults to surface_gap; "
        "use signed_penetration only for historical WarpSDF comparisons."
    ),
)
parser.add_argument("--pressure-stiffness", type=float, default=None)
parser.add_argument("--pressure-damping", type=float, default=None)
parser.add_argument("--pressure-max-force", type=float, default=None)
parser.add_argument("--pressure-gain", type=float, default=1.0)
parser.add_argument("--pressure-bias", type=float, default=0.0)
parser.add_argument("--pressure-gamma", type=float, default=1.0)
parser.add_argument("--pressure-threshold", type=float, default=0.0)
parser.add_argument("--pressure-calib", type=str, default=None)
parser.add_argument("--pressure-layout-urdf", type=str, default=None)
parser.add_argument("--pressure-layout-link", type=str, default=None)
parser.add_argument("--pressure-layout-origin-l", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--taxel-area", type=float, default=1.0)
parser.add_argument("--penetration-deadband", type=float, default=0.0)
parser.add_argument(
    "--mesh-shell-thickness",
    type=float,
    default=None,
    help="Surface-gap pressure sensing shell thickness in meters.",
)
parser.add_argument(
    "--pressure-surface-gap-mode",
    choices=("normal_ray", "nearest"),
    default="normal_ray",
    help="How surface_gap measures mesh clearance: taxel normal ray, or legacy nearest-point distance.",
)
parser.add_argument(
    "--pressure-response-model",
    choices=("auto", "penetration_kv", "gap_fraction"),
    default="auto",
    help=(
        "Pressure response after contact geometry. auto uses gap_fraction for surface_gap and "
        "penetration_kv for signed_penetration."
    ),
)
parser.add_argument("--enable-normal-ray-pressure", action="store_true")
parser.add_argument("--normal-ray-pressure-deadband", type=float, default=0.0)
parser.add_argument("--normal-ray-pressure-max-distance", type=float, default=0.00075)
parser.add_argument("--normal-ray-pressure-distance-mode", choices=("plane", "local3d"), default="plane")
parser.add_argument("--enable-geometry-normal-ray-pressure", action="store_true")
parser.add_argument("--geometry-normal-ray-usd-path", type=str, default=None)
parser.add_argument("--geometry-normal-ray-vertices-npy", type=str, default=None)
parser.add_argument("--geometry-normal-ray-triangles-npy", type=str, default=None)
parser.add_argument("--geometry-normal-ray-mode", choices=("inside_exit", "baseline"), default="inside_exit")
parser.add_argument("--geometry-normal-ray-origin-source", choices=("pressure_taxel", "tacmap_aligned"), default="pressure_taxel")
parser.add_argument("--geometry-normal-ray-deadband", type=float, default=0.0)
parser.add_argument("--geometry-normal-ray-max-distance", type=float, default=0.02)
parser.add_argument("--geometry-normal-ray-taxel-samples", choices=("center", "cross_5", "grid_3x3"), default="center")
parser.add_argument("--geometry-normal-ray-sample-spacing", type=float, default=None)
parser.add_argument("--geometry-normal-ray-sample-aggregation", choices=("max", "mean", "positive_mean"), default="max")
parser.add_argument("--geometry-normal-ray-sample-min-support-fraction", type=float, default=0.0)
parser.add_argument(
    "--geometry-normal-ray-rest-distance",
    type=float,
    default=None,
    help="Constant rest distance in meters. Defaults to first-frame ray-hit baseline.",
)
parser.add_argument(
    "--mesh-unsigned-shell-as-contact",
    action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--enable-physx-contact-map",
    action="store_true",
    help="Enable sparse PhysX contact projection as a sanity/debug channel, not dense pressure GT.",
)
parser.add_argument("--disable-physx-contact-map", action="store_true")
parser.add_argument("--physx-contact-max-data", type=int, default=64)
parser.add_argument("--physx-contact-kernel-sigma", type=float, default=None)
parser.add_argument("--physx-contact-kernel-radius", type=float, default=None)
parser.add_argument(
    "--enable-warpsdf-contact-gate",
    action="store_true",
    help="Experimental/debug: gate WarpSDF output using sparse PhysX contacts. Do not use as L1 acceptance GT.",
)
parser.add_argument("--disable-warpsdf-contact-gate", action="store_true")
parser.add_argument("--warpsdf-contact-gate-min-contacts", type=int, default=1)
parser.add_argument("--tacmap-gamma", type=float, default=1.0)
parser.add_argument("--tacmap-ray-mode", choices=("surface_normal", "link_surface"), default="surface_normal")
parser.add_argument("--link-surface-ray-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+x")
parser.add_argument("--link-surface-ray-direction", type=float, nargs=3, default=None)
parser.add_argument("--link-surface-use-axis-direction", action="store_true")
parser.add_argument("--link-surface-grid-u-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+y")
parser.add_argument("--link-surface-grid-v-axis", choices=("+x", "-x", "+y", "-y", "+z", "-z"), default="+z")
parser.add_argument("--link-surface-grid-u-size", type=float, default=0.014)
parser.add_argument("--link-surface-grid-v-size", type=float, default=0.020)
parser.add_argument("--link-surface-grid-center", type=float, nargs=3, default=(-0.008, 0.0, 0.0012))
parser.add_argument("--link-surface-ray-viz-length", type=float, default=0.008)
parser.add_argument("--link-surface-ray-viz-width", type=float, default=0.00006)
parser.add_argument("--tacmap-view", choices=("raw", "quantized", "mask", "normalized"), default="quantized")
parser.add_argument("--tacmap-display-max-mm", type=float, default=None)
parser.add_argument("--hide-tacmap-center-arrow", action="store_true")
parser.add_argument("--tacmap-center-arrow-length", type=float, default=0.018)
parser.add_argument("--tacmap-center-arrow-width", type=float, default=0.00045)
parser.add_argument("--tacmap-center-arrow-head-length", type=float, default=0.004)
parser.add_argument("--tacmap-center-arrow-head-width", type=float, default=0.0012)
parser.add_argument("--tacmap-center-arrow-center-radius", type=float, default=0.0012)
parser.add_argument("--tacmap-center-arrow-start-offset", type=float, default=0.0015)
parser.add_argument("--tacmap-center-arrow-sigma-px", type=float, default=10.0)
parser.add_argument("--tacmap-center-arrow-flip", action="store_true")
parser.add_argument("--hide-tacmap-finger-center-arrows", action="store_true")
parser.add_argument("--tacmap-finger-center-arrow-length", type=float, default=0.012)
parser.add_argument("--tacmap-finger-center-arrow-width", type=float, default=0.00035)
parser.add_argument("--tacmap-finger-center-arrow-head-length", type=float, default=0.003)
parser.add_argument("--tacmap-finger-center-arrow-head-width", type=float, default=0.001)
parser.add_argument("--tacmap-finger-center-arrow-center-radius", type=float, default=0.001)
parser.add_argument("--tacmap-normal-force-stiffness", type=float, default=None)
parser.add_argument("--tacmap-normal-force-damping", type=float, default=None)
parser.add_argument("--tacmap-normal-force-arrow-scale", type=float, default=1.0e-4)
parser.add_argument("--tacmap-normal-force-min-arrow-length", type=float, default=0.003)
parser.add_argument("--tacmap-normal-force-sample-area", type=float, default=1.0)
parser.add_argument("--active-threshold", type=float, default=1.0e-4)
parser.add_argument("--live_web", action="store_true")
parser.add_argument("--live_port", type=int, default=8090)
parser.add_argument("--show_cv", action="store_true")
parser.add_argument(
    "--set-viewport-camera",
    dest="set_viewport_camera",
    action="store_true",
    default=None,
    help="Set the Isaac Sim viewport camera to the tactile press scene on startup.",
)
parser.add_argument(
    "--no-set-viewport-camera",
    dest="set_viewport_camera",
    action="store_false",
    help="Do not modify the Isaac Sim viewport camera on startup.",
)
parser.add_argument("--viewport-camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--viewport-camera-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--save_maps", action="store_true")
parser.add_argument("--save_heightmaps", action="store_true")
parser.add_argument("--save-pressure-trace", action="store_true")
parser.add_argument("--pressure-trace-dir", type=str, default=str(REPO_ROOT / "outputs" / "pressure_traces"))
parser.add_argument("--verify-pressure-trace", action="store_true", help="Run pressure trace verifier after saving the trace.")
parser.add_argument("--pressure-verify-out", type=str, default=None, help="Verifier JSON path or output directory.")
parser.add_argument("--pressure-verify-pressure-key", type=str, default="pressure_norm")
parser.add_argument("--pressure-verify-raw-key", type=str, default="pressure_raw_n")
parser.add_argument("--pressure-verify-penetration-key", type=str, default="penetration_m")
parser.add_argument(
    "--pressure-verify-reference-key",
    type=str,
    default=None,
    help="Optional model/benchmark reference key in the trace, e.g. tacmap_raw_m or analytic_depth_m.",
)
parser.add_argument(
    "--pressure-verify-reference-valid-mask-key",
    type=str,
    default=None,
    help="Optional alignment valid-mask key. Defaults to '<reference-key>_valid_mask' when present.",
)
parser.add_argument(
    "--pressure-verify-reference-layer",
    type=str,
    default=None,
    help=(
        "Optional validation layer metadata: L0_analytic, L1_model_reference, "
        "L1_sparse_sanity, L2_offline_oracle, L3_real_calibration, or benchmark_reference. "
        "Inferred from --pressure-verify-reference-key when omitted."
    ),
)
parser.add_argument(
    "--pressure-verify-dense-reference-layer",
    action="append",
    default=None,
    help=(
        "Reference layer allowed to drive dense spatial acceptance. Can be repeated. "
        f"Defaults to {', '.join(DEFAULT_DENSE_REFERENCE_ACCEPTANCE_LAYERS)}."
    ),
)
parser.add_argument(
    "--pressure-verify-allow-reference-layer-override",
    action="store_true",
    help=(
        "Allow --pressure-verify-reference-layer to override an inferred layer. "
        "Intended only for deliberate diagnostics."
    ),
)
parser.add_argument("--pressure-verify-active-threshold", type=float, default=1.0e-6)
parser.add_argument("--pressure-verify-penetration-threshold", type=float, default=0.0)
parser.add_argument("--pressure-verify-reference-threshold", type=float, default=0.0)
parser.add_argument("--pressure-verify-precontact-leakage-threshold", type=float, default=0.001)
parser.add_argument("--pressure-verify-reference-iou-threshold", type=float, default=0.95)
parser.add_argument("--pressure-verify-reference-alignment-valid-fraction-threshold", type=float, default=None)
parser.add_argument("--pressure-verify-reference-contact-valid-fraction-threshold", type=float, default=1.0)
parser.add_argument("--pressure-verify-reference-invalid-contact-fraction-threshold", type=float, default=0.0)
parser.add_argument("--pressure-verify-centroid-threshold-px", type=float, default=1.5)
parser.add_argument("--pressure-verify-bbox-threshold-px", type=float, default=2.0)
parser.add_argument("--pressure-verify-depth-rmse-threshold-m", type=float, default=2.0e-5)
parser.add_argument("--pressure-verify-onset-threshold-frames", type=int, default=1)
parser.add_argument("--pressure-verify-offset-threshold-frames", type=int, default=1)
parser.add_argument("--pressure-verify-spearman-threshold", type=float, default=0.95)
parser.add_argument("--pressure-verify-fail-on-threshold", action="store_true")
parser.add_argument("--enable_fots", action="store_true")
parser.add_argument("--save_fots", action="store_true")
parser.add_argument("--fots-contact-threshold-mm", type=float, default=0.02)
parser.add_argument("--fots-depth-scale", type=float, default=1.0)
parser.add_argument("--fots-lamb-dilate", type=float, default=0.00125)
parser.add_argument("--fots-lamb-shear", type=float, default=0.00021)
parser.add_argument("--fots-lamb-twist", type=float, default=0.00038)
parser.add_argument("--fots-theta", type=float, default=0.0)
parser.add_argument("--fots-track-contact-center", action="store_true")
parser.add_argument("--fots-depth-background", action="store_true")
parser.add_argument("--fots-depth-background-max-mm", type=float, default=None)
parser.add_argument("--fots-arrow-scale", type=float, default=1.0)
parser.add_argument("--fots-marker-size", type=float, default=3.0)
parser.add_argument("--fots-marker-size-depth-gain", type=float, default=2.0)
parser.add_argument("--fots-motion-contact-radius-px", type=float, default=0.0)
parser.add_argument("--fots-show-noncontact-motion", action="store_true")
parser.add_argument("--fots-view", choices=("flow", "markers", "both"), default="markers")
parser.add_argument("--enable_tacex_rgb", action="store_true")
parser.add_argument("--save_tacex_rgb", action="store_true")
parser.add_argument("--tacex-rgb-width", type=int, default=320)
parser.add_argument("--tacex-rgb-height", type=int, default=240)
parser.add_argument("--tacex-rgb-depth-scale", type=float, default=1.0)
parser.add_argument("--tacex-rgb-with-shadow", action="store_true")
parser.add_argument("--tacex-rgb-device", type=str, default=None)
parser.add_argument(
    "--tacex-rgb-calib-dir",
    type=str,
    default=str(DEFAULT_TACEX_RGB_CALIB_DIR),
)
parser.add_argument(
    "--tacex-rgb-sim-dir",
    type=str,
    default=str(DEFAULT_TACEX_RGB_SIM_DIR),
)
parser.add_argument("--enable_tacsl_shear", "--enable_tacsl_force_field", action="store_true", dest="enable_tacsl_shear")
parser.add_argument("--save_tacsl_shear", action="store_true")
parser.add_argument("--tacsl-sdf-gradient-eps", type=float, default=1.0e-4)
parser.add_argument("--tacsl-normal-contact-stiffness", type=float, default=1.0)
parser.add_argument("--tacsl-tangential-stiffness", type=float, default=0.1)
parser.add_argument("--tacsl-friction-coefficient", type=float, default=2.0)
parser.add_argument("--tacsl-shear-normal-force-threshold", type=float, default=0.00008)
parser.add_argument("--tacsl-shear-force-threshold", type=float, default=0.001)
parser.add_argument("--tacsl-shear-resolution", type=int, default=8)
parser.add_argument("--tacsl-shear-render-rows", type=int, default=9)
parser.add_argument("--tacsl-shear-render-cols", type=int, default=11)
parser.add_argument("--tacsl-shear-render-stride", type=int, default=0)
parser.add_argument("--tacsl-shear-min-arrow-length-px", type=float, default=0.0)
parser.add_argument("--enable_hydroshear_marker", action="store_true")
parser.add_argument("--save_hydroshear_marker", action="store_true")
parser.add_argument(
    "--hydroshear-debug-only",
    "--hydroshear_debug_only",
    action="store_true",
    dest="hydroshear_debug_only",
)
parser.add_argument("--hydroshear-debug-visuals", "--hydroshear_debug_visuals", action="store_true")
parser.add_argument("--show-hydroshear-sample-points", "--show_hydroshear_sample_points", action="store_true")
parser.add_argument("--show-hydroshear-marker-points", "--show_hydroshear_marker_points", action="store_true")
parser.add_argument("--show-hydroshear-marker-normals", "--show_hydroshear_marker_normals", action="store_true")
parser.add_argument("--show-hydroshear-marker-axes", "--show_hydroshear_marker_axes", action="store_true")
parser.add_argument(
    "--show-tacmap-finger-marker-points",
    "--show_tacmap_finger_marker_points",
    action="store_true",
    dest="show_tacmap_finger_marker_points",
)
parser.add_argument("--hydroshear-sample-point-radius", type=float, default=0.00045)
parser.add_argument("--hydroshear-marker-point-radius", type=float, default=0.00055)
parser.add_argument("--hydroshear-marker-normal-length", type=float, default=0.003)
parser.add_argument("--hydroshear-marker-normal-width", type=float, default=0.00008)
parser.add_argument("--hydroshear-marker-axis-length", type=float, default=0.002)
parser.add_argument("--hydroshear-marker-axis-width", type=float, default=0.00006)
parser.add_argument("--hydroshear-marker-axis-max", type=int, default=1000)
parser.add_argument("--hydroshear-sample-point-max", type=int, default=5000)
parser.add_argument("--hydroshear-contact-threshold-mm", type=float, default=0.02)
parser.add_argument("--hydroshear-lamb-dilate", type=float, default=20000.0)
parser.add_argument("--hydroshear-lamb-shear", type=float, default=25000.0)
parser.add_argument("--hydroshear-dilate-scale", type=float, default=30.0)
parser.add_argument("--hydroshear-shear-scale", type=float, default=50.0)
parser.add_argument("--hydroshear-hydrosoft-k", type=float, default=1.0)
parser.add_argument("--hydroshear-hydrosoft-e", type=float, default=1.0)
parser.add_argument("--hydroshear-hydrosoft-area", type=float, default=1.0)
parser.add_argument("--hydroshear-hydrosoft-mu", type=float, default=0.5)
parser.add_argument("--hydroshear-arrow-scale", type=float, default=24000.0)
parser.add_argument("--hydroshear-min-arrow-length-px", type=float, default=0.0)
parser.add_argument("--hydroshear-depth-background", action="store_true")
parser.add_argument("--hydroshear-depth-background-max-mm", type=float, default=8.0)
parser.add_argument("--hydroshear-object-sample-mode", choices=("random", "poisson"), default="poisson")
parser.add_argument("--hydroshear-object-sample-count", type=int, default=32768)
parser.add_argument("--hydroshear-poisson-radius", type=float, default=0.00075)
parser.add_argument("--hydroshear-poisson-initial-count", type=int, default=1000000)
parser.add_argument("--hydroshear-object-sample-seed", type=int, default=17)
parser.add_argument("--save_every", type=int, default=30)
parser.add_argument("--print-every", type=int, default=30)
parser.add_argument("--contact-benchmark-out", type=str, default=None)
parser.add_argument("--contact-benchmark-sensor", type=int, default=0)
parser.add_argument("--contact-benchmark-force-threshold", type=float, default=1.0e-4)
parser.add_argument("--contact-benchmark-depth-threshold-mm", type=float, default=0.001)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.hydroshear_debug_only or args.hydroshear_debug_visuals:
    args.enable_hydroshear_marker = True
    args.show_hydroshear_sample_points = True
    args.show_hydroshear_marker_points = True
    args.show_hydroshear_marker_normals = True
    args.show_hydroshear_marker_axes = True
    args.show_tacmap_finger_marker_points = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from integrated_tactile_env import IntegratedTactileEnv, IntegratedTactileEnvCfg  # noqa: E402
from aloha_tactile_env import pressure_sensor_labels_from_env, press_depth_summary_from_env  # noqa: E402
from curved_hydroshear_adapter import RevoCurvedHydroShearAdapter, RevoCurvedHydroShearCfg  # noqa: E402
from fots_adapter import RevoFotsAdapter, RevoFotsCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from tacex_rgb_adapter import RevoTacExRgbAdapter, RevoTacExRgbCfg  # noqa: E402
from BrainCo_DexHand.force_map import (  # noqa: E402
    GeometryNormalRayPenetrationSource,
    NormalRayPenetrationSource,
    PressureCalibration,
    aligned_reference_points_for_pressure_taxels,
    apply_reference_alignment_to_values,
    build_reference_to_pressure_alignment,
    calibrate_penetration,
    evaluate_pressure_trace_report,
    load_pressure_calibration_overrides,
    load_pressure_pad_specs_from_urdf,
    pressure_calibration_value_summary,
    pressure_map_stats,
    pressure_trace_report,
    select_pressure_pad_spec,
    triangle_mesh_topology_diagnostics,
    validate_pressure_trace_v1,
    weld_duplicate_triangle_vertices,
)
from tacsl_shear_adapter import RevoTacslShearAdapter, RevoTacslShearCfg  # noqa: E402


SENSOR_LABELS = ["thumb", "index", "middle", "ring", "pinky"]
_REVO21_TACMAP_DIR = TACMAP_ROOT / "assets" / "tactilesensor_map" / "revo21_dv2"
_REVO21_VITAI_MARKER_LAYOUT = (
    REPO_ROOT
    / "assets"
    / "revo21_right_touch"
    / "marker_positions"
    / "vitai_4fingers"
    / "marker_positions.npz"
)


def _finger_map(
    stem: str,
    *,
    grid_u_size: float,
    grid_v_size: float,
    grid_center: tuple[float, float, float],
    ray_axis: str = "+x",
    grid_u_axis: str = "+y",
    grid_v_axis: str = "+z",
) -> dict[str, object]:
    return {
        "touch_link": f"{stem}_link",
        "points": _REVO21_TACMAP_DIR / f"{stem}_point.npy",
        "normals": _REVO21_TACMAP_DIR / f"{stem}_normal.npy",
        "link_surface": {
            "ray_axis": ray_axis,
            "grid_u_axis": grid_u_axis,
            "grid_v_axis": grid_v_axis,
            "grid_u_size": float(grid_u_size),
            "grid_v_size": float(grid_v_size),
            "grid_center": tuple(float(value) for value in grid_center),
        },
    }


FINGER_MAPS = {
    "middle": _finger_map(
        "right_middip_roll_rubber",
        grid_u_size=0.0202,
        grid_v_size=0.0182,
        grid_center=(0.00159, 0.00103, 0.04757),
    ),
    "index": _finger_map(
        "right_indexdip_roll_rubber",
        grid_u_size=0.02068358,
        grid_v_size=0.01846460,
        grid_center=(0.00105538, 0.00114480, 0.01765266),
    ),
    "ring": _finger_map(
        "right_ringdip_roll_rubber",
        grid_u_size=0.02056884,
        grid_v_size=0.01818433,
        grid_center=(0.00104103, -0.00038099, 0.01768645),
    ),
    "pinky": _finger_map(
        "right_pinkydip_roll_rubber",
        grid_u_size=0.02068569,
        grid_v_size=0.01843646,
        grid_center=(0.00100531, -0.00114829, 0.01764967),
    ),
    "thumb": _finger_map(
        "right_thumbdip_roll_rubber",
        ray_axis="+y",
        grid_u_axis="+x",
        grid_v_axis="+z",
        grid_u_size=0.01343826,
        grid_v_size=0.02319049,
        grid_center=(-0.00072565, 0.02421566, -0.00022462),
    ),
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
        "default_extra_rot_deg": -110.425,
        "rot": (
            1.0,
            1.4563845496008064e-15,
            -1.5916065870379613e-16,
            5.5510727714786566e-17,
        ),
    },
    "ball_probe": {
        "usd": REPO_ROOT / "tacmap" / "assets" / "presser" / "ball_probe.usd",
        "default_flip_axis": "none",
        "default_extra_rot_axis": "x",
        "default_extra_rot_deg": 90.0,
        "rot": (1.0, 0.0, 0.0, 0.0),
    },
}

_LIVE_LOCK = threading.Lock()
_LIVE_IMAGE_RGB: np.ndarray | None = None
_LIVE_STATS = "waiting for integrated tactile frames..."


def _jet_colormap(values) -> torch.Tensor:
    return _jet_colormap_t(torch.as_tensor(values, dtype=torch.float32))


def _normalize_grid(grid, gamma: float) -> torch.Tensor:
    return _normalize_grid_t(grid, gamma)


def force_strip(tactile, *, scale: int, gamma: float) -> torch.Tensor:
    return _force_strip_t(tactile, scale=scale, gamma=gamma)


def _fixed_scale_grid(grid, vmax: float, gamma: float) -> torch.Tensor:
    return _fixed_scale_grid_t(grid, vmax, gamma)


def _depth_weighted_center_px(depth) -> tuple[float, float] | None:
    return _depth_weighted_center_px_t(depth)


def _draw_ring(
    img,
    center_xy: tuple[float, float],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 7,
    thickness: int = 2,
) -> None:
    from tensor_image_utils import draw_ring as _draw_ring_t

    _draw_ring_t(img, center_xy, color=color, radius=radius, thickness=thickness)


def _normalize_vec(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1.0e-9:
        return None
    return (arr / norm).astype(np.float32)


def _quat_from_x_axis(direction: np.ndarray) -> np.ndarray:
    target = _normalize_vec(direction)
    if target is None:
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    source = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    if dot < -1.0 + 1.0e-8:
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    xyz = np.cross(source, target)
    quat = np.array((1.0 + dot, xyz[0], xyz[1], xyz[2]), dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-9)
    return quat


def _estimated_normal_force_center_w(
    depth: np.ndarray,
    prev_depth: np.ndarray | None,
    surface_points_w: np.ndarray | None,
    surface_normals_w: np.ndarray | None,
    surface_valid: np.ndarray | None,
    *,
    dt: float,
    stiffness: float,
    damping: float,
    sample_area: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    depth = np.maximum(np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if surface_points_w is None or surface_normals_w is None:
        return None

    points = np.asarray(surface_points_w, dtype=np.float32)
    normals = np.asarray(surface_normals_w, dtype=np.float32)
    if points.shape[:2] != depth.shape or normals.shape[:2] != depth.shape or points.shape[-1] != 3 or normals.shape[-1] != 3:
        return None

    if prev_depth is None:
        depth_velocity = np.zeros_like(depth, dtype=np.float32)
    else:
        prev = np.maximum(np.nan_to_num(np.asarray(prev_depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        depth_velocity = np.zeros_like(depth, dtype=np.float32) if prev.shape != depth.shape else (depth - prev) / max(float(dt), 1.0e-9)

    if surface_valid is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = np.asarray(surface_valid, dtype=bool)
        if valid.shape != depth.shape:
            valid = np.reshape(valid, depth.shape)

    normal_norm = np.linalg.norm(normals, axis=-1)
    valid = (
        valid
        & np.isfinite(depth)
        & np.isfinite(depth_velocity)
        & np.isfinite(points).all(axis=-1)
        & np.isfinite(normals).all(axis=-1)
        & (normal_norm > 1.0e-9)
    )
    unit_normals = normals / np.maximum(normal_norm[..., None], 1.0e-9)

    local_force = float(stiffness) * depth + float(damping) * depth_velocity
    local_force = np.maximum(local_force, 0.0) * max(float(sample_area), 0.0)
    local_force = np.where(valid & (depth > 0.0), local_force, 0.0).astype(np.float32)
    force_total = float(local_force.sum())
    if force_total <= 1.0e-12:
        return None

    center_w = np.sum(local_force[..., None] * points, axis=(0, 1)) / force_total
    force_w = np.sum(local_force[..., None] * unit_normals, axis=(0, 1))
    force_norm = float(np.linalg.norm(force_w))
    if not np.isfinite(force_norm) or force_norm <= 1.0e-12:
        return None
    return center_w.astype(np.float32), force_w.astype(np.float32), force_norm


def tacmap_strip(
    tacmap,
    *,
    tacmap_raw=None,
    scale: int,
    gamma: float,
    view: str,
    display_max_m: float,
) -> torch.Tensor:
    # Semantics kept from the original NumPy implementation:
    # pressure_center = (_depth_weighted_center_px(tacmap_raw[i]) ...)
    # _draw_ring(
    return _tacmap_strip_t(
        tacmap,
        tacmap_raw=tacmap_raw,
        scale=scale,
        gamma=gamma,
        view=view,
        display_max_m=display_max_m,
    )


def image_strip(images, *, scale: int = 1) -> np.ndarray:
    return _image_strip_t(images, scale=scale)


def sensor_display_batch(values, sensor_index: int | None):
    if values is None:
        return None
    arr = values if isinstance(values, torch.Tensor) else np.asarray(values)
    if sensor_index is None or arr.ndim == 0 or arr.shape[0] <= 0:
        return arr
    idx = max(0, min(int(sensor_index), int(arr.shape[0]) - 1))
    return arr[idx : idx + 1]


def _env_torch_device(env) -> torch.device:
    fallback = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(getattr(env, "_device", fallback))


def _finite_device_tensor(value, env, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=_env_torch_device(env), dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype, device=_env_torch_device(env))
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0) if tensor.is_floating_point() else tensor


def pressure_force_map_tensor_from_env(env, fallback=None) -> torch.Tensor | None:
    sensors = getattr(env, "_tactile_sensors", None)
    cfg = getattr(env, "_cfg", None)
    if not sensors or cfg is None:
        return None
    rows = max(1, int(getattr(cfg, "num_rows", 1)))
    cols = max(1, int(getattr(cfg, "num_cols", 1)))
    if fallback is not None and hasattr(fallback, "shape") and len(fallback.shape) >= 3:
        slot_count = int(fallback.shape[0])
    else:
        slot_order = getattr(env, "_sensor_slot_order", [])
        slot_count = max(len(sensors), max(slot_order) + 1 if slot_order else 0)
    out = torch.zeros((slot_count, rows, cols), dtype=torch.float32, device=_env_torch_device(env))
    slot_order = getattr(env, "_sensor_slot_order", list(range(len(sensors))))
    for sensor_index, sensor in enumerate(sensors):
        slot = int(slot_order[sensor_index]) if sensor_index < len(slot_order) else sensor_index
        if not (0 <= slot < slot_count):
            continue
        sensor_data = getattr(sensor, "data", None)
        if sensor_data is None:
            continue
        source = None
        force_map = getattr(sensor_data, "pressure_force_map", None)
        if force_map is not None:
            source = force_map.detach()
            if source.ndim >= 4:
                source = source[0, 0]
            elif source.ndim == 3:
                source = source[0]
        else:
            tactile_points = getattr(sensor_data, "tactile_points_w", None)
            if tactile_points is not None and tactile_points.shape[1] == rows * cols:
                source = tactile_points[0, :, 3].detach().reshape(rows, cols)
        if source is None or source.ndim < 2:
            continue
        source = _finite_device_tensor(source[:rows, :cols], env)
        out[slot, : source.shape[0], : source.shape[1]] = source
    return out


def tacmap_raw_tensor_from_env(env) -> torch.Tensor | None:
    sensors = getattr(env, "_tacmap_sensors", None)
    if not sensors:
        return None
    rows = int(getattr(env, "_tacmap_rows", 0))
    cols = int(getattr(env, "_tacmap_cols", 0))
    if rows <= 0 or cols <= 0:
        return None
    output_shape_fn = getattr(env, "_tacmap_output_shape", None)
    shape = tuple(output_shape_fn()) if callable(output_shape_fn) else (len(sensors), rows, cols)
    out = torch.zeros(shape, dtype=torch.float32, device=_env_torch_device(env))
    raw_tensor_fn = getattr(env, "_tacmap_raw_tensor", None)
    slot_order = getattr(env, "_tacmap_slot_order", list(range(len(sensors))))
    for sensor_index, _sensor in enumerate(sensors):
        slot = int(slot_order[sensor_index]) if sensor_index < len(slot_order) else sensor_index
        if not (0 <= slot < out.shape[0]) or not callable(raw_tensor_fn):
            continue
        source = raw_tensor_fn(sensor_index, rows, cols)
        if source is None:
            continue
        source = _finite_device_tensor(source, env)
        out[slot, : source.shape[0], : source.shape[1]] = source
    return out


def tacmap_surface_raw_tensor_from_env(env) -> torch.Tensor | None:
    surface_sensors = getattr(env, "_tacmap_surface_sensors", None)
    if not surface_sensors:
        return None
    rows = int(getattr(env, "_tacmap_rows", 0))
    cols = int(getattr(env, "_tacmap_cols", 0))
    if rows <= 0 or cols <= 0:
        return None
    output_shape_fn = getattr(env, "_tacmap_output_shape", None)
    shape = tuple(output_shape_fn()) if callable(output_shape_fn) else (len(surface_sensors), rows, cols)
    out = torch.zeros(shape, dtype=torch.float32, device=_env_torch_device(env))
    surface_raw_fn = getattr(env, "_surface_raw_tensor", None)
    slot_order = getattr(env, "_tacmap_slot_order", list(range(len(surface_sensors))))
    for sensor_index, _sensor in enumerate(surface_sensors):
        slot = int(slot_order[sensor_index]) if sensor_index < len(slot_order) else sensor_index
        if not (0 <= slot < out.shape[0]) or not callable(surface_raw_fn):
            continue
        source = surface_raw_fn(sensor_index, rows, cols)
        if source is None:
            continue
        source = _finite_device_tensor(source, env)
        out[slot, : source.shape[0], : source.shape[1]] = source
    return out


def tacmap_link_surface_geometry_tensors_from_env(env):
    sensors = getattr(env, "_tacmap_sensors", None)
    if not sensors:
        return None
    rows = int(getattr(env, "_tacmap_rows", 0))
    cols = int(getattr(env, "_tacmap_cols", 0))
    if rows <= 0 or cols <= 0:
        return None
    output_shape_fn = getattr(env, "_tacmap_output_shape", None)
    shape = tuple(output_shape_fn()) if callable(output_shape_fn) else (len(sensors), rows, cols)
    device = _env_torch_device(env)
    surface_points = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
    surface_normals = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
    object_points = torch.zeros((*shape, 3), dtype=torch.float32, device=device)
    surface_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    object_valid = torch.zeros(shape, dtype=torch.bool, device=device)
    slot_order = getattr(env, "_tacmap_slot_order", list(range(len(sensors))))
    surface_sensors = getattr(env, "_tacmap_surface_sensors", [])
    raw_tensor_fn = getattr(env, "_raw_sensor_tensor", None)
    num_points = rows * cols
    for sensor_index, sensor in enumerate(sensors):
        slot = int(slot_order[sensor_index]) if sensor_index < len(slot_order) else sensor_index
        if not (0 <= slot < shape[0]):
            continue
        surface_sensor = surface_sensors[sensor_index] if sensor_index < len(surface_sensors) else None
        if surface_sensor is not None:
            hits = getattr(surface_sensor, "second_ray_hits_w", None)
            normals = getattr(surface_sensor, "second_ray_normals_w", None)
            valid = getattr(surface_sensor, "second_ray_hit_valid", None)
            if hits is not None and valid is not None and hits.shape[1] == num_points:
                surface_points[slot] = _finite_device_tensor(hits[0].detach().reshape(rows, cols, 3), env)
                surface_valid[slot] = valid[0].detach().to(device=device, dtype=torch.bool).reshape(rows, cols)
            if normals is not None and normals.shape[1] == num_points:
                surface_normals[slot] = _finite_device_tensor(normals[0].detach().reshape(rows, cols, 3), env)
        hits = getattr(sensor, "ray_hits_w", None)
        if hits is not None and hits.shape[1] == num_points:
            object_points[slot] = _finite_device_tensor(hits[0].detach().reshape(rows, cols, 3), env)
        if callable(raw_tensor_fn):
            raw = raw_tensor_fn(sensor, rows, cols)
            if raw is not None:
                object_valid[slot] = (_finite_device_tensor(raw, env) > 0.0).reshape(rows, cols)
    return surface_points, surface_normals, surface_valid, object_points, object_valid


def _resize_nearest(img, height: int) -> np.ndarray:
    return _resize_nearest_t(img, height)


def _resize_nearest_to_shape(img, height: int, width: int) -> np.ndarray:
    return _resize_nearest_to_shape_t(img, height, width)


def combined_image(*panes) -> np.ndarray:
    return _combined_image_t(*panes)


def _grid_active_summary(grid: np.ndarray, *, threshold: float) -> str:
    values = np.nan_to_num(np.asarray(grid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    active_mask = values > float(threshold)
    active = int(np.count_nonzero(active_mask))
    if active <= 0:
        return f"max:{float(values.max()):.4f} active:0/{values.size}"

    positive = np.where(active_mask, values, 0.0)
    total = float(positive.sum())
    rows, cols = np.indices(values.shape)
    if total > 0.0:
        row_c = float((rows * positive).sum() / total)
        col_c = float((cols * positive).sum() / total)
    else:
        active_rows, active_cols = np.nonzero(active_mask)
        row_c = float(np.mean(active_rows))
        col_c = float(np.mean(active_cols))
    active_rows, active_cols = np.nonzero(active_mask)
    bbox = (
        int(active_rows.min()),
        int(active_cols.min()),
        int(active_rows.max()),
        int(active_cols.max()),
    )
    peak_flat = int(np.argmax(values))
    peak_row, peak_col = np.unravel_index(peak_flat, values.shape)
    return (
        f"max:{float(values.max()):.4f} active:{active}/{values.size} "
        f"cop:({row_c:.1f},{col_c:.1f}) peak:({int(peak_row)},{int(peak_col)}) "
        f"bbox:({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})"
    )



def hydroshear_debug_image(hydroshear_output, *, sensor_index: int | None = None) -> np.ndarray | None:
    if hydroshear_output is None:
        return None
    return combined_image(
        image_strip(sensor_display_batch(hydroshear_output.debug_marker_displacement_images, sensor_index)),
        image_strip(sensor_display_batch(hydroshear_output.debug_sdf_images, sensor_index)),
        image_strip(sensor_display_batch(hydroshear_output.debug_projection_images, sensor_index)),
        image_strip(sensor_display_batch(hydroshear_output.debug_mdilate_images, sensor_index)),
    )


class UsdPointInstancerViz:
    """Small USD point renderer shared by tactile debug overlays."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        radius: float,
        proto_name: str,
        parent_paths: tuple[str, ...] = (),
    ):
        self.path = str(path)
        self.color = tuple(float(v) for v in color)
        self.radius = max(1.0e-6, float(radius))
        self.proto_name = _usd_safe_component(proto_name)
        self.parent_paths = tuple(str(value) for value in parent_paths)
        self._inst = None
        self._gf = None
        self._vt = None

    def set_points(self, points_w: np.ndarray) -> None:
        self._ensure_stage_objects()
        gf = self._gf
        vt = self._vt
        pts = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
        self._inst.GetPositionsAttr().Set(vt.Vec3fArray([gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts]))
        self._inst.GetProtoIndicesAttr().Set(vt.IntArray([0] * int(pts.shape[0])))
        self._inst.GetScalesAttr().Set(vt.Vec3fArray([gf.Vec3f(1.0, 1.0, 1.0)] * int(pts.shape[0])))

    def _ensure_stage_objects(self) -> None:
        if self._inst is not None:
            return
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        path = Sdf.Path(self.path)
        for parent_path in self.parent_paths:
            UsdGeom.Xform.Define(stage, Sdf.Path(parent_path))
        UsdGeom.Xform.Define(stage, path.GetParentPath())
        inst = UsdGeom.PointInstancer.Define(stage, path)
        inst.CreatePositionsAttr()
        inst.CreateProtoIndicesAttr()
        inst.CreateScalesAttr()

        proto_root = path.AppendPath("Prototypes")
        UsdGeom.Xform.Define(stage, proto_root)
        sphere_path = proto_root.AppendPath(self.proto_name)
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(self.radius)
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(self.color[0], self.color[1], self.color[2])])
        )

        inst.CreatePrototypesRel().SetTargets([sphere_path])
        self._inst = inst
        self._gf = Gf
        self._vt = Vt


class UsdSegmentCurveViz:
    """Small USD line-segment renderer shared by vector debug overlays."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        width: float,
        parent_paths: tuple[str, ...] = (),
    ):
        self.path = str(path)
        self.color = tuple(float(v) for v in color)
        self.width = max(1.0e-6, float(width))
        self.parent_paths = tuple(str(value) for value in parent_paths)
        self._curves = None
        self._gf = None
        self._vt = None

    def set_segments(self, starts_w: np.ndarray, ends_w: np.ndarray) -> None:
        self._ensure_stage_objects()
        gf = self._gf
        vt = self._vt
        starts = np.asarray(starts_w, dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(ends_w, dtype=np.float32).reshape(-1, 3)
        count = min(starts.shape[0], ends.shape[0])
        points = []
        for start, end in zip(starts[:count], ends[:count]):
            points.append(gf.Vec3f(float(start[0]), float(start[1]), float(start[2])))
            points.append(gf.Vec3f(float(end[0]), float(end[1]), float(end[2])))
        self._curves.GetCurveVertexCountsAttr().Set(vt.IntArray([2] * int(count)))
        self._curves.GetPointsAttr().Set(vt.Vec3fArray(points))
        self._curves.GetWidthsAttr().Set(vt.FloatArray([self.width] * int(2 * count)))

    def _ensure_stage_objects(self) -> None:
        if self._curves is not None:
            return
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, Vt

        stage = omni.usd.get_context().get_stage()
        path = Sdf.Path(self.path)
        for parent_path in self.parent_paths:
            UsdGeom.Xform.Define(stage, Sdf.Path(parent_path))
        UsdGeom.Xform.Define(stage, path.GetParentPath())
        curves = UsdGeom.BasisCurves.Define(stage, path)
        curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr()
        curves.CreatePointsAttr()
        curves.CreateWidthsAttr()
        UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(self.color[0], self.color[1], self.color[2])])
        )
        self._curves = curves
        self._gf = Gf
        self._vt = Vt


class ArrowMarkerViz:
    """Shared three-primitive arrow renderer: center dot, shaft, and head."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        width: float,
        head_width: float,
        head_length: float,
        center_radius: float,
    ):
        self.path = str(path)
        self.color = tuple(float(v) for v in color)
        self.width = max(1.0e-6, float(width))
        self.head_width = max(0.0, float(head_width))
        self.head_length = max(0.0, float(head_length))
        self.center_radius = max(1.0e-6, float(center_radius))
        self._markers: VisualizationMarkers | None = None

    def visualize(
        self,
        *,
        centers_w: np.ndarray,
        starts_w: np.ndarray,
        directions_w: np.ndarray,
        lengths_m: np.ndarray,
        max_length: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_markers()
        centers = np.asarray(centers_w, dtype=np.float32).reshape(-1, 3)
        starts = np.asarray(starts_w, dtype=np.float32).reshape(-1, 3)
        directions = np.asarray(directions_w, dtype=np.float32).reshape(-1, 3)
        lengths = np.asarray(lengths_m, dtype=np.float32).reshape(-1)
        count = min(centers.shape[0], starts.shape[0], directions.shape[0], lengths.shape[0])
        translations = []
        orientations = []
        scales = []
        marker_indices = []
        valid_starts = []
        valid_dirs = []
        valid_lengths = []
        shaft_radius = max(1.0e-6, self.width)
        head_radius = max(shaft_radius, self.head_width)
        clamp_length = None if max_length is None else max(1.0e-6, float(max_length))
        for center, start, direction, arrow_length in zip(
            centers[:count], starts[:count], directions[:count], lengths[:count]
        ):
            unit = _normalize_vec(direction)
            arrow_length = float(arrow_length)
            if unit is None or not np.isfinite(arrow_length) or arrow_length <= 0.0:
                continue
            arrow_length = max(arrow_length, 1.0e-6)
            if clamp_length is not None:
                arrow_length = min(arrow_length, clamp_length)
            head_len = min(self.head_length, max(0.0, 0.55 * arrow_length))
            shaft_len = max(1.0e-6, arrow_length - head_len)
            quat = _quat_from_x_axis(unit)
            translations.extend((center, start + unit * (0.5 * shaft_len), start + unit * (shaft_len + 0.5 * head_len)))
            orientations.extend((quat, quat, quat))
            scales.extend(
                (
                    (self.center_radius, self.center_radius, self.center_radius),
                    (shaft_len, shaft_radius, shaft_radius),
                    (head_len, head_radius, head_radius),
                )
            )
            marker_indices.extend((0, 1, 2))
            valid_starts.append(start)
            valid_dirs.append(unit)
            valid_lengths.append(arrow_length)

        if not translations:
            self._markers.set_visibility(False)
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        self._markers.set_visibility(True)
        self._markers.visualize(
            translations=np.asarray(translations, dtype=np.float32),
            orientations=np.asarray(orientations, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            marker_indices=np.asarray(marker_indices, dtype=np.int32),
        )
        return (
            np.asarray(valid_starts, dtype=np.float32).reshape(-1, 3),
            np.asarray(valid_dirs, dtype=np.float32).reshape(-1, 3),
            np.asarray(valid_lengths, dtype=np.float32).reshape(-1),
        )

    def visualize_paths(
        self,
        *,
        centers_w: np.ndarray,
        starts_w: np.ndarray,
        ends_w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = np.asarray(starts_w, dtype=np.float32).reshape(-1, 3)
        ends = np.asarray(ends_w, dtype=np.float32).reshape(-1, 3)
        count = min(starts.shape[0], ends.shape[0])
        deltas = ends[:count] - starts[:count]
        lengths = np.linalg.norm(deltas, axis=-1).astype(np.float32)
        return self.visualize(
            centers_w=np.asarray(centers_w, dtype=np.float32).reshape(-1, 3)[:count],
            starts_w=starts[:count],
            directions_w=deltas,
            lengths_m=lengths,
        )

    def _ensure_markers(self) -> None:
        if self._markers is not None:
            return
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=self.color, roughness=0.5)
        cfg = VisualizationMarkersCfg(
            prim_path=self.path,
            markers={
                "center": sim_utils.SphereCfg(radius=1.0, visual_material=material),
                "shaft": sim_utils.CylinderCfg(radius=1.0, height=1.0, axis="X", visual_material=material),
                "head": sim_utils.ConeCfg(radius=1.0, height=1.0, axis="X", visual_material=material),
            },
        )
        self._markers = VisualizationMarkers(cfg)


class HydroShearPointViz:
    """Display HydroShear debug point sets in the USD stage."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        primary_attr: str,
        fallback_attr: str | None = None,
        radius: float,
        max_points: int,
        sensor_index: int | None = None,
    ):
        self.primary_attr = str(primary_attr)
        self.fallback_attr = None if fallback_attr is None else str(fallback_attr)
        self.max_points = max(1, int(max_points))
        self.sensor_index = None if sensor_index is None else int(sensor_index)
        self._points = UsdPointInstancerViz(
            path=str(path),
            color=tuple(float(v) for v in color),
            radius=radius,
            proto_name="SamplePoint",
        )

    def update(self, hydroshear_output) -> None:
        if hydroshear_output is None:
            self._set_points(np.zeros((0, 3), dtype=np.float32))
            return
        point_sets = getattr(hydroshear_output, self.primary_attr, None)
        if not point_sets and self.fallback_attr is not None:
            point_sets = getattr(hydroshear_output, self.fallback_attr, None)
        if not point_sets:
            self._set_points(np.zeros((0, 3), dtype=np.float32))
            return
        if self.sensor_index is not None:
            idx = max(0, min(int(self.sensor_index), len(point_sets) - 1))
            point_sets = point_sets[idx : idx + 1]
        points = []
        for point_set in point_sets:
            arr = np.asarray(point_set, dtype=np.float32).reshape(-1, 3)
            if arr.size:
                points.append(arr)
        if not points:
            self._set_points(np.zeros((0, 3), dtype=np.float32))
            return
        pts = np.concatenate(points, axis=0)
        finite = np.isfinite(pts).all(axis=-1)
        pts = pts[finite]
        if pts.shape[0] > self.max_points:
            stride = max(1, int(np.ceil(float(pts.shape[0]) / float(self.max_points))))
            pts = pts[::stride][: self.max_points]
        self._set_points(pts.astype(np.float32))

    def _set_points(self, points_w: np.ndarray) -> None:
        self._points.set_points(points_w)


def _usd_safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _tacmap_point_grid(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    arr = arr.reshape(-1, 3)
    side = int(round(math.sqrt(float(arr.shape[0]))))
    if side * side != arr.shape[0]:
        raise ValueError(f"TacMap point map must be a square grid or HxWx3 array, got {arr.shape}")
    return arr.reshape(side, side, 3)


_TACMAP_NPY_CACHE: dict[str, np.ndarray] = {}


def _load_tacmap_npy(path: Path | str) -> np.ndarray:
    key = str(Path(path))
    cached = _TACMAP_NPY_CACHE.get(key)
    if cached is None:
        cached = np.load(Path(path))
        _TACMAP_NPY_CACHE[key] = cached
    return cached


def _hydroshear_marker_pixel_indices(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    marker_cfg = RevoCurvedHydroShearCfg(width=int(width), height=int(height))
    xs = np.linspace(
        float(marker_cfg.marker_margin_x),
        float(width) - float(marker_cfg.marker_margin_x),
        int(marker_cfg.marker_cols),
    )
    ys = np.linspace(
        float(marker_cfg.marker_margin_y),
        float(height) - float(marker_cfg.marker_margin_y),
        int(marker_cfg.marker_rows),
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    ix = np.clip(np.rint(grid_x.reshape(-1)).astype(np.int32), 0, int(width) - 1)
    iy = np.clip(np.rint(grid_y.reshape(-1)).astype(np.int32), 0, int(height) - 1)
    return ix, iy


class TacMapFingerLinkVizBase:
    """Shared robot-link lookup for per-finger TacMap debug overlays."""

    def __init__(self) -> None:
        self._entries: list[dict[str, object]] = []
        self._body_names_signature: tuple[str, ...] | None = None
        self._body_indices_by_link: dict[str, int | None] = {}
        self._warned_links: set[str] = set()

    def _body_link_state(self, env) -> tuple[tuple[str, ...], np.ndarray] | None:
        robot = getattr(env, "_robot", None)
        state = _tensor_row_to_numpy(getattr(getattr(robot, "data", None), "body_link_state_w", None), ndim=3)
        if robot is None or state is None:
            return None
        return tuple(str(name) for name in getattr(robot, "body_names", [])), state

    def _refresh_body_indices(self, body_names: tuple[str, ...]) -> None:
        if self._body_names_signature == body_names:
            return
        self._body_names_signature = body_names
        self._body_indices_by_link = {}
        for entry in self._entries:
            link_name = str(entry["link_name"])
            if link_name in body_names:
                self._body_indices_by_link[link_name] = body_names.index(link_name)
                continue
            matches = [
                i
                for i, body_name in enumerate(body_names)
                if body_name.endswith(link_name) or link_name in body_name
            ]
            self._body_indices_by_link[link_name] = matches[0] if len(matches) == 1 else None

    def _entry_pose(self, entry: dict[str, object], state: np.ndarray, warn_label: str) -> np.ndarray | None:
        link_name = str(entry["link_name"])
        body_idx = self._body_indices_by_link.get(link_name)
        if body_idx is None or body_idx >= state.shape[0]:
            if link_name not in self._warned_links:
                print(f"[WARN] {warn_label} link not found in robot bodies: {link_name}", flush=True)
                self._warned_links.add(link_name)
            return None
        pose = np.asarray(state[int(body_idx), :7], dtype=np.float32)
        return pose if _pose_valid(pose) else None


class TacMapFingerMarkerPointViz(TacMapFingerLinkVizBase):
    """Display calibrated Vitai marker positions on each fingertip rubber link."""

    def __init__(
        self,
        *,
        fingers: tuple[str, ...],
        root_path: str,
        color: tuple[float, float, float],
        radius: float,
    ):
        super().__init__()
        self.root_path = str(root_path).rstrip("/")
        self.color = tuple(float(v) for v in color)
        self.radius = max(1.0e-6, float(radius))

        try:
            with np.load(_REVO21_VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
                distortion_valid = np.asarray(layout["distortion_valid"], dtype=bool)
                for finger in fingers:
                    spec = FINGER_MAPS.get(finger)
                    points_key = f"{finger}_points_link_m"
                    normals_key = f"{finger}_normals_link"
                    method_key = f"{finger}_method"
                    if (
                        spec is None
                        or points_key not in layout
                        or normals_key not in layout
                        or method_key not in layout
                    ):
                        continue
                    link_name = str(spec["touch_link"])
                    marker_points_l = np.asarray(layout[points_key], dtype=np.float32).reshape(-1, 3)
                    marker_normals_l = np.asarray(layout[normals_key], dtype=np.float32).reshape(-1, 3)
                    methods = np.asarray(layout[method_key]).astype(str).reshape(-1)
                    normal_norm = np.linalg.norm(marker_normals_l, axis=-1)
                    finite = (
                        np.isfinite(marker_points_l).all(axis=-1)
                        & (np.linalg.norm(marker_points_l, axis=-1) > 1.0e-9)
                        & np.isfinite(marker_normals_l).all(axis=-1)
                        & (normal_norm > 1.0e-9)
                    )
                    exact = finite & distortion_valid & (methods == "ray_hit")
                    points = marker_points_l[exact]
                    if not len(points):
                        continue
                    prim_name = (
                        f"{_usd_safe_component(finger)}_{_usd_safe_component(link_name)}_"
                        "MarkerPointsBlue"
                    )
                    self._entries.append(
                        {
                            "finger": str(finger),
                            "link_name": link_name,
                            "quality": "high_confidence",
                            "points_l": points.astype(np.float32),
                            "viz": UsdPointInstancerViz(
                                path=f"{self.root_path}/{prim_name}",
                                color=self.color,
                                radius=self.radius,
                                proto_name="MarkerPoint",
                                parent_paths=("/Visuals", self.root_path),
                            ),
                        }
                    )
        except Exception as exc:
            print(
                f"[WARN] calibrated Vitai marker viz disabled: "
                f"cannot load {_REVO21_VITAI_MARKER_LAYOUT}: {exc}",
                flush=True,
            )

    def update(self, env) -> None:
        if not self._entries:
            return
        state_info = self._body_link_state(env)
        if state_info is None:
            for entry in self._entries:
                self._set_points(entry, np.zeros((0, 3), dtype=np.float32))
            return

        body_names, state = state_info
        self._refresh_body_indices(body_names)
        for entry in self._entries:
            pose = self._entry_pose(entry, state, "TacMap finger marker viz")
            if pose is None:
                self._set_points(entry, np.zeros((0, 3), dtype=np.float32))
                continue
            points_l = np.asarray(entry["points_l"], dtype=np.float32).reshape(-1, 3)
            points_w = pose[:3][None, :] + quat_apply_np(pose[3:7], points_l)
            self._set_points(entry, points_w.astype(np.float32))

    def _set_points(self, entry: dict[str, object], points_w: np.ndarray) -> None:
        entry["viz"].set_points(points_w)


class TacMapFingerMarkerNormalViz(TacMapFingerLinkVizBase):
    """Display calibrated Vitai marker surface normals as yellow line segments."""

    def __init__(
        self,
        *,
        fingers: tuple[str, ...],
        root_path: str,
        color: tuple[float, float, float],
        length: float,
        width: float,
    ):
        super().__init__()
        self.root_path = str(root_path).rstrip("/")
        self.color = tuple(float(v) for v in color)
        self.length = max(0.0, float(length))
        self.width = max(1.0e-6, float(width))

        try:
            with np.load(_REVO21_VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
                distortion_valid = np.asarray(layout["distortion_valid"], dtype=bool).reshape(-1)
                for finger in fingers:
                    spec = FINGER_MAPS.get(finger)
                    points_key = f"{finger}_points_link_m"
                    normals_key = f"{finger}_normals_link"
                    method_key = f"{finger}_method"
                    if (
                        spec is None
                        or points_key not in layout
                        or normals_key not in layout
                        or method_key not in layout
                    ):
                        continue
                    link_name = str(spec["touch_link"])
                    marker_points_l = np.asarray(layout[points_key], dtype=np.float32).reshape(-1, 3)
                    marker_normals_l = np.asarray(layout[normals_key], dtype=np.float32).reshape(-1, 3)
                    methods = np.asarray(layout[method_key]).astype(str).reshape(-1)
                    if not (
                        len(marker_points_l)
                        == len(marker_normals_l)
                        == len(methods)
                        == len(distortion_valid)
                    ):
                        raise ValueError(f"Calibrated marker array length mismatch for {finger}")
                    normal_norm = np.linalg.norm(marker_normals_l, axis=-1)
                    valid = (
                        np.isfinite(marker_points_l).all(axis=-1)
                        & (np.linalg.norm(marker_points_l, axis=-1) > 1.0e-9)
                        & np.isfinite(marker_normals_l).all(axis=-1)
                        & (normal_norm > 1.0e-9)
                    )
                    exact = valid & distortion_valid & (methods == "ray_hit")
                    marker_points_l = marker_points_l[exact]
                    marker_normals_l = marker_normals_l[exact] / normal_norm[exact, None]
                    prim_name = (
                        f"{_usd_safe_component(finger)}_{_usd_safe_component(link_name)}_"
                        "MarkerNormalsYellow"
                    )
                    self._entries.append(
                        {
                            "finger": str(finger),
                            "link_name": link_name,
                            "points_l": marker_points_l.astype(np.float32),
                            "normals_l": marker_normals_l.astype(np.float32),
                            "viz": UsdSegmentCurveViz(
                                path=f"{self.root_path}/{prim_name}",
                                color=self.color,
                                width=self.width,
                                parent_paths=("/Visuals", self.root_path),
                            ),
                        }
                    )
        except Exception as exc:
            print(
                f"[WARN] calibrated Vitai marker normal viz disabled: "
                f"cannot load {_REVO21_VITAI_MARKER_LAYOUT}: {exc}",
                flush=True,
            )

    def update(self, env) -> None:
        if not self._entries:
            return
        state_info = self._body_link_state(env)
        if state_info is None or self.length <= 0.0:
            for entry in self._entries:
                self._set_segments(entry, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return

        body_names, state = state_info
        self._refresh_body_indices(body_names)
        for entry in self._entries:
            pose = self._entry_pose(entry, state, "TacMap finger marker normal viz")
            if pose is None:
                self._set_segments(entry, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
                continue
            points_l = np.asarray(entry["points_l"], dtype=np.float32).reshape(-1, 3)
            normals_l = np.asarray(entry["normals_l"], dtype=np.float32).reshape(-1, 3)
            count = min(points_l.shape[0], normals_l.shape[0])
            points_w = pose[:3][None, :] + quat_apply_np(pose[3:7], points_l[:count])
            normals_w = quat_apply_np(pose[3:7], normals_l[:count])
            normal_norm = np.linalg.norm(normals_w, axis=-1)
            valid = np.isfinite(points_w).all(axis=-1) & np.isfinite(normals_w).all(axis=-1) & (normal_norm > 1.0e-9)
            starts = points_w[valid].astype(np.float32)
            dirs = (normals_w[valid] / normal_norm[valid, None]).astype(np.float32)
            self._set_segments(entry, starts, (starts + dirs * self.length).astype(np.float32))

    def _set_segments(self, entry: dict[str, object], starts_w: np.ndarray, ends_w: np.ndarray) -> None:
        entry["viz"].set_segments(starts_w, ends_w)


class TacMapFingerCenterArrowViz(TacMapFingerLinkVizBase):
    """Display each selected finger's link_surface center and scripted press path as green arrows."""

    def __init__(
        self,
        *,
        fingers: tuple[str, ...],
        path: str,
        color: tuple[float, float, float],
        length: float,
        width: float,
        head_length: float,
        head_width: float,
        center_radius: float,
        use_mean_normal: bool,
        custom_ray_direction: tuple[float, float, float] | None,
        press_start_offset: float,
        press_end_offset: float,
    ):
        super().__init__()
        self.length = max(0.0, float(length))
        self.use_mean_normal = bool(use_mean_normal)
        self.custom_ray_direction = custom_ray_direction
        self.press_start_offset = float(press_start_offset)
        self.press_end_offset = float(press_end_offset)
        self._arrows = ArrowMarkerViz(
            path=str(path),
            color=tuple(float(v) for v in color),
            width=width,
            head_width=head_width,
            head_length=head_length,
            center_radius=center_radius,
        )

        for finger in fingers:
            spec = FINGER_MAPS.get(finger)
            if spec is None:
                continue
            link_surface = spec.get("link_surface")
            if not isinstance(link_surface, dict):
                continue
            center_l = np.asarray(link_surface.get("grid_center", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3)
            direction_l = self._resolve_direction_l(spec, link_surface)
            if direction_l is None:
                print(f"[WARN] TacMap finger center arrow skipped {finger}: no valid ray direction.", flush=True)
                continue
            self._entries.append(
                {
                    "finger": str(finger),
                    "link_name": str(spec["touch_link"]),
                    "center_l": center_l.astype(np.float32),
                    "direction_l": direction_l.astype(np.float32),
                }
            )
            print(
                "[INFO] TacMap finger center arrow: "
                f"finger={finger}, link={spec['touch_link']}, "
                f"center_l=({center_l[0]:.6g},{center_l[1]:.6g},{center_l[2]:.6g}), "
                f"normal_l=({direction_l[0]:.3f},{direction_l[1]:.3f},{direction_l[2]:.3f}), "
                f"path_offsets={self.press_start_offset:.6g}->{self.press_end_offset:.6g}",
                flush=True,
            )

    def _resolve_direction_l(self, spec: dict[str, object], link_surface: dict[str, object]) -> np.ndarray | None:
        if self.custom_ray_direction is not None:
            direction = _normalize_vec(np.asarray(self.custom_ray_direction, dtype=np.float32))
            if direction is not None:
                return direction

        if self.use_mean_normal:
            normals_path = Path(spec["normals"])
            try:
                normals = _load_tacmap_npy(normals_path).reshape(-1, 3)
            except Exception as exc:
                print(
                    f"[WARN] TacMap finger center arrow cannot load mean normal from {normals_path}: {exc}",
                    flush=True,
                )
            else:
                norms = np.linalg.norm(normals, axis=-1)
                valid = np.isfinite(normals).all(axis=-1) & (norms > 1.0e-8)
                if np.any(valid):
                    normalized = normals[valid] / norms[valid, None]
                    direction = _normalize_vec(np.mean(normalized, axis=0))
                    if direction is not None:
                        return direction

        return _normalize_vec(np.asarray(axis_to_vector(str(link_surface.get("ray_axis", "+x"))), dtype=np.float32))

    def update(self, env) -> None:
        if not self._entries or self.length <= 0.0:
            self._set_arrows(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            return

        state_info = self._body_link_state(env)
        if state_info is None:
            self._set_arrows(
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
            return

        body_names, state = state_info
        self._refresh_body_indices(body_names)
        centers = []
        starts = []
        ends = []
        for entry in self._entries:
            pose = self._entry_pose(entry, state, "TacMap finger center arrow")
            if pose is None:
                continue
            center_l = np.asarray(entry["center_l"], dtype=np.float32).reshape(1, 3)
            direction_l = np.asarray(entry["direction_l"], dtype=np.float32).reshape(1, 3)
            center_w = pose[:3] + quat_apply_np(pose[3:7], center_l)[0]
            direction_w = _normalize_vec(quat_apply_np(pose[3:7], direction_l)[0])
            if direction_w is None or not np.all(np.isfinite(center_w)):
                continue
            path_start_w = center_w + direction_w * self.press_start_offset
            path_end_w = center_w + direction_w * self.press_end_offset
            centers.append(center_w.astype(np.float32))
            starts.append(path_start_w.astype(np.float32))
            ends.append(path_end_w.astype(np.float32))
        self._set_arrows(
            np.asarray(centers, dtype=np.float32).reshape(-1, 3),
            np.asarray(starts, dtype=np.float32).reshape(-1, 3),
            np.asarray(ends, dtype=np.float32).reshape(-1, 3),
        )

    def _set_arrows(self, centers_w: np.ndarray, starts_w: np.ndarray, ends_w: np.ndarray) -> None:
        self._arrows.visualize_paths(centers_w=centers_w, starts_w=starts_w, ends_w=ends_w)


class HydroShearVectorViz:
    """Display HydroShear debug vectors as short colored USD curve segments."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        points_attr: str,
        vectors_attr: str,
        length: float,
        width: float,
        max_vectors: int,
        sensor_index: int | None = None,
    ):
        self.points_attr = str(points_attr)
        self.vectors_attr = str(vectors_attr)
        self.length = max(0.0, float(length))
        self.max_vectors = max(1, int(max_vectors))
        self.sensor_index = None if sensor_index is None else int(sensor_index)
        self._segments = UsdSegmentCurveViz(
            path=str(path),
            color=tuple(float(v) for v in color),
            width=width,
        )

    def update(self, hydroshear_output) -> None:
        if hydroshear_output is None or self.length <= 0.0:
            self._set_segments(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return

        point_sets = getattr(hydroshear_output, self.points_attr, None)
        vector_sets = getattr(hydroshear_output, self.vectors_attr, None)
        if not point_sets or not vector_sets:
            self._set_segments(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return
        if self.sensor_index is not None:
            count = min(len(point_sets), len(vector_sets))
            idx = max(0, min(int(self.sensor_index), count - 1))
            point_sets = point_sets[idx : idx + 1]
            vector_sets = vector_sets[idx : idx + 1]

        starts = []
        vectors = []
        for point_set, vector_set in zip(point_sets, vector_sets):
            pts = np.asarray(point_set, dtype=np.float32).reshape(-1, 3)
            vec = np.asarray(vector_set, dtype=np.float32).reshape(-1, 3)
            count = min(pts.shape[0], vec.shape[0])
            if count:
                starts.append(pts[:count])
                vectors.append(vec[:count])
        if not starts:
            self._set_segments(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return

        pts = np.concatenate(starts, axis=0)
        vec = np.concatenate(vectors, axis=0)
        vec_norm = np.linalg.norm(vec, axis=-1)
        finite = np.isfinite(pts).all(axis=-1) & np.isfinite(vec).all(axis=-1) & (vec_norm > 1.0e-9)
        pts = pts[finite]
        vec = vec[finite] / vec_norm[finite, None]
        if pts.shape[0] > self.max_vectors:
            stride = max(1, int(np.ceil(float(pts.shape[0]) / float(self.max_vectors))))
            pts = pts[::stride][: self.max_vectors]
            vec = vec[::stride][: self.max_vectors]
        self._set_segments(pts.astype(np.float32), (pts + vec * self.length).astype(np.float32))

    def _set_segments(self, starts_w: np.ndarray, ends_w: np.ndarray) -> None:
        self._segments.set_segments(starts_w, ends_w)


class TacMapCenterArrowViz:
    """Display TacMap depth-derived center normal force estimates as red 3D marker arrows."""

    def __init__(
        self,
        *,
        path: str,
        color: tuple[float, float, float],
        length: float,
        width: float,
        head_length: float,
        head_width: float,
        center_radius: float,
        start_offset: float,
        sigma_px: float,
        flip: bool,
        dt: float,
        stiffness: float,
        damping: float,
        force_arrow_scale: float,
        min_arrow_length: float,
        sample_area: float,
    ):
        self.length = max(0.0, float(length))
        self.start_offset = float(start_offset)
        self.sigma_px = max(1.0, float(sigma_px))
        self.flip = bool(flip)
        self.dt = max(float(dt), 1.0e-9)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.force_arrow_scale = max(0.0, float(force_arrow_scale))
        self.min_arrow_length = max(0.0, float(min_arrow_length))
        self.sample_area = max(0.0, float(sample_area))
        self._arrows = ArrowMarkerViz(
            path=str(path),
            color=tuple(float(v) for v in color),
            width=width,
            head_width=head_width,
            head_length=head_length,
            center_radius=center_radius,
        )
        self._prev_depth: np.ndarray | None = None
        self._last_starts = np.zeros((0, 3), dtype=np.float32)
        self._last_dirs = np.zeros((0, 3), dtype=np.float32)
        self._last_lengths = np.zeros((0,), dtype=np.float32)
        self._last_force_norms = np.zeros((0,), dtype=np.float32)
        self._last_force_vectors = np.zeros((0, 3), dtype=np.float32)

    def update(
        self,
        tacmap_raw: np.ndarray | None,
        surface_points_w: np.ndarray | None,
        surface_normals_w: np.ndarray | None,
        surface_valid: np.ndarray | None,
    ) -> None:
        if tacmap_raw is None or surface_points_w is None or surface_normals_w is None or self.length <= 0.0:
            self._prev_depth = None
            self._set_arrows(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return

        raw = np.asarray(tacmap_raw, dtype=np.float32)
        points = np.asarray(surface_points_w, dtype=np.float32)
        normals = np.asarray(surface_normals_w, dtype=np.float32)
        valid = None if surface_valid is None else np.asarray(surface_valid, dtype=bool)
        if raw.ndim != 3 or points.ndim != 4 or normals.ndim != 4:
            self._prev_depth = None
            self._set_arrows(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
            return

        starts = []
        dirs = []
        lengths = []
        force_norms = []
        force_vectors = []
        prev_depth = self._prev_depth if self._prev_depth is not None and self._prev_depth.shape == raw.shape else None
        count = min(raw.shape[0], points.shape[0], normals.shape[0])
        for i in range(count):
            item_valid = valid[i] if valid is not None and i < valid.shape[0] else None
            estimate = _estimated_normal_force_center_w(
                raw[i],
                prev_depth[i] if prev_depth is not None and i < prev_depth.shape[0] else None,
                points[i],
                normals[i],
                item_valid,
                dt=self.dt,
                stiffness=self.stiffness,
                damping=self.damping,
                sample_area=self.sample_area,
            )
            if estimate is None:
                continue
            center_w, force_w, force_norm = estimate
            unit = _normalize_vec(-force_w if self.flip else force_w)
            if unit is None:
                continue
            starts.append(center_w + unit * self.start_offset)
            dirs.append(unit)
            lengths.append(
                min(self.length, max(self.min_arrow_length, force_norm * self.force_arrow_scale))
                if self.force_arrow_scale > 0.0
                else self.length
            )
            force_norms.append(force_norm)
            force_vectors.append(force_w)

        self._prev_depth = raw.copy()
        self._set_arrows(
            np.asarray(starts, dtype=np.float32).reshape(-1, 3),
            np.asarray(dirs, dtype=np.float32).reshape(-1, 3),
            lengths_m=np.asarray(lengths, dtype=np.float32).reshape(-1),
            force_norms=np.asarray(force_norms, dtype=np.float32).reshape(-1),
            force_vectors=np.asarray(force_vectors, dtype=np.float32).reshape(-1, 3),
        )

    def stats_line(self, labels: list[str]) -> str:
        if self._last_starts.shape[0] == 0:
            return "normal_force_arrow=off"
        parts = []
        for i, (start, direction, force_norm) in enumerate(
            zip(self._last_starts, self._last_dirs, self._last_force_norms)
        ):
            label = labels[i] if i < len(labels) else str(i)
            parts.append(
                f"{label}:p=({start[0]:.4f},{start[1]:.4f},{start[2]:.4f}) "
                f"n=({direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}) "
                f"Fn_est={force_norm:.3g}"
            )
        return "normal_force_arrow=on " + " ".join(parts)

    def _set_arrows(
        self,
        starts_w: np.ndarray,
        directions_w: np.ndarray,
        *,
        lengths_m: np.ndarray | None = None,
        force_norms: np.ndarray | None = None,
        force_vectors: np.ndarray | None = None,
    ) -> None:
        starts = np.asarray(starts_w, dtype=np.float32).reshape(-1, 3)
        directions = np.asarray(directions_w, dtype=np.float32).reshape(-1, 3)
        count = min(starts.shape[0], directions.shape[0])
        lengths = (
            np.asarray(lengths_m, dtype=np.float32).reshape(-1)
            if lengths_m is not None
            else np.full((count,), self.length, dtype=np.float32)
        )
        norms = (
            np.asarray(force_norms, dtype=np.float32).reshape(-1)
            if force_norms is not None
            else np.zeros((count,), dtype=np.float32)
        )
        vectors = (
            np.asarray(force_vectors, dtype=np.float32).reshape(-1, 3)
            if force_vectors is not None
            else np.zeros((count, 3), dtype=np.float32)
        )
        count = min(count, lengths.shape[0], norms.shape[0], vectors.shape[0])
        last_starts = []
        last_dirs = []
        last_lengths = []
        last_force_norms = []
        last_force_vectors = []
        for start, direction, arrow_length, force_norm, force_vector in zip(
            starts[:count], directions[:count], lengths[:count], norms[:count], vectors[:count]
        ):
            unit = _normalize_vec(direction)
            arrow_length = float(arrow_length)
            if unit is None or not np.isfinite(arrow_length) or arrow_length <= 0.0:
                continue
            arrow_length = min(max(arrow_length, 1.0e-6), self.length)
            last_starts.append(start)
            last_dirs.append(unit)
            last_lengths.append(arrow_length)
            last_force_norms.append(float(force_norm))
            last_force_vectors.append(force_vector)

        self._last_starts = np.asarray(last_starts, dtype=np.float32).reshape(-1, 3)
        self._last_dirs = np.asarray(last_dirs, dtype=np.float32).reshape(-1, 3)
        self._last_lengths = np.asarray(last_lengths, dtype=np.float32).reshape(-1)
        self._last_force_norms = np.asarray(last_force_norms, dtype=np.float32).reshape(-1)
        self._last_force_vectors = np.asarray(last_force_vectors, dtype=np.float32).reshape(-1, 3)
        self._arrows.visualize(
            centers_w=self._last_starts,
            starts_w=self._last_starts,
            directions_w=self._last_dirs,
            lengths_m=self._last_lengths,
            max_length=self.length,
        )


def poisson_disk_downsample_points(points: np.ndarray, *, radius: float, seed: int) -> np.ndarray:
    """Approximate Poisson disk downsampling for fixed object surface samples."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=-1)
    points = points[finite]
    if points.size == 0 or float(radius) <= 0.0:
        return points.astype(np.float32)

    try:
        import point_cloud_utils as pcu

        idx = pcu.downsample_point_cloud_poisson_disk(points, radius=float(radius), target_num_samples=-1)
        return points[np.asarray(idx, dtype=np.int64)].astype(np.float32)
    except Exception:
        pass

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(points.shape[0])
    ordered = points[order]
    cell_size = float(radius)
    coords = np.floor(ordered / cell_size).astype(np.int64)

    # First keep one random candidate per radius-sized voxel to make the later
    # neighbor pass cheap even when the initial candidate cloud is very dense.
    _, first = np.unique(coords, axis=0, return_index=True)
    first = np.sort(first)
    candidates = ordered[first]

    selected: list[np.ndarray] = []
    grid: dict[tuple[int, int, int], list[int]] = {}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    radius2 = float(radius) * float(radius)

    for point in candidates:
        cell = tuple(np.floor(point / cell_size).astype(np.int64).tolist())
        keep = True
        for offset in offsets:
            neighbor = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            for selected_idx in grid.get(neighbor, []):
                delta = point - selected[selected_idx]
                if float(np.dot(delta, delta)) < radius2:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            grid.setdefault(cell, []).append(len(selected))
            selected.append(point)

    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def hydroshear_candidate_sample_count(*, sample_count: int, sample_mode: str, poisson_initial_count: int) -> int:
    return int(poisson_initial_count) if str(sample_mode) == "poisson" else int(sample_count)


def maybe_poisson_downsample_hydroshear_samples(
    samples: np.ndarray,
    *,
    sample_mode: str,
    poisson_radius: float,
    seed: int,
    label: str,
) -> np.ndarray:
    if str(sample_mode) != "poisson":
        return samples
    before = int(np.asarray(samples).shape[0])
    downsampled = poisson_disk_downsample_points(samples, radius=float(poisson_radius), seed=int(seed))
    print(
        f"[INFO] HydroShear Poisson downsampled {label}: {before} -> {downsampled.shape[0]} "
        f"(radius={float(poisson_radius):.6g}m)",
        flush=True,
    )
    return downsampled


def sample_usd_mesh_surface_points(
    usd_path: str | Path,
    *,
    sample_count: int,
    seed: int,
    scale: float = 1.0,
    sample_mode: str = "random",
    poisson_radius: float = 0.00075,
    poisson_initial_count: int = 1000000,
) -> np.ndarray | None:
    """Sample fixed points over all mesh faces in a USD, returned in root-local coordinates."""
    candidate_count = hydroshear_candidate_sample_count(
        sample_count=sample_count,
        sample_mode=sample_mode,
        poisson_initial_count=poisson_initial_count,
    )
    if candidate_count <= 0:
        return None

    try:
        from pxr import Gf, Usd, UsdGeom
    except Exception as exc:
        print(f"[WARN] HydroShear mesh sampling disabled: pxr is unavailable: {exc}", flush=True)
        return None

    usd_path = Path(usd_path).expanduser().resolve()
    if not usd_path.is_file():
        print(f"[WARN] HydroShear mesh sampling disabled: missing USD {usd_path}", flush=True)
        return None

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print(f"[WARN] HydroShear mesh sampling disabled: could not open USD {usd_path}", flush=True)
        return None

    root_prim = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    try:
        root_inv = cache.GetLocalToWorldTransform(root_prim).GetInverse() if root_prim and root_prim.IsValid() else Gf.Matrix4d(1.0)
    except Exception:
        root_inv = Gf.Matrix4d(1.0)

    all_vertices: list[np.ndarray] = []
    all_triangles: list[np.ndarray] = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue

        mesh_to_world = cache.GetLocalToWorldTransform(prim)
        vertices = []
        for p in points:
            p_root = root_inv.Transform(mesh_to_world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))))
            vertices.append((float(p_root[0]) * scale, float(p_root[1]) * scale, float(p_root[2]) * scale))
        vertices = np.asarray(vertices, dtype=np.float64)

        local_tris = []
        cursor = 0
        face_indices = np.asarray(indices, dtype=np.int64)
        for count in np.asarray(counts, dtype=np.int64):
            face = face_indices[cursor : cursor + int(count)]
            cursor += int(count)
            if face.size < 3:
                continue
            for j in range(1, face.size - 1):
                local_tris.append([int(face[0]), int(face[j]), int(face[j + 1])])

        if not local_tris:
            continue
        all_vertices.append(vertices)
        all_triangles.append(np.asarray(local_tris, dtype=np.int64) + vertex_offset)
        vertex_offset += vertices.shape[0]

    if not all_vertices or not all_triangles:
        print(f"[WARN] HydroShear mesh sampling disabled: no mesh triangles in {usd_path}", flush=True)
        return None

    vertices = np.concatenate(all_vertices, axis=0)
    triangles = np.concatenate(all_triangles, axis=0)
    tri_vertices = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 1.0e-18)
    if not np.any(valid):
        print(f"[WARN] HydroShear mesh sampling disabled: zero-area mesh in {usd_path}", flush=True)
        return None

    tri_vertices = tri_vertices[valid]
    areas = areas[valid]
    probs = areas / np.sum(areas)
    rng = np.random.default_rng(int(seed))
    tri_ids = rng.choice(tri_vertices.shape[0], size=candidate_count, replace=True, p=probs)
    chosen = tri_vertices[tri_ids]
    r1 = rng.random(candidate_count)
    r2 = rng.random(candidate_count)
    sqrt_r1 = np.sqrt(r1)
    samples = (
        (1.0 - sqrt_r1)[:, None] * chosen[:, 0]
        + (sqrt_r1 * (1.0 - r2))[:, None] * chosen[:, 1]
        + (sqrt_r1 * r2)[:, None] * chosen[:, 2]
    )
    samples = samples.astype(np.float32)
    samples = maybe_poisson_downsample_hydroshear_samples(
        samples,
        sample_mode=sample_mode,
        poisson_radius=poisson_radius,
        seed=seed,
        label="source-USD samples",
    )
    print(
        f"[INFO] HydroShear object mesh samples: {samples.shape[0]} points from {usd_path.name} "
        f"over {tri_vertices.shape[0]} triangles",
        flush=True,
    )
    return samples


def sample_stage_object_surface_points(
    prim_path: str,
    *,
    sample_count: int,
    seed: int,
    sample_mode: str = "random",
    poisson_radius: float = 0.00075,
    poisson_initial_count: int = 1000000,
) -> np.ndarray | None:
    """Use IsaacLab's object point-cloud sampler on the already-spawned stage object."""
    candidate_count = hydroshear_candidate_sample_count(
        sample_count=sample_count,
        sample_mode=sample_mode,
        poisson_initial_count=poisson_initial_count,
    )
    if candidate_count <= 0:
        return None
    try:
        import torch
        from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.utils import sample_object_point_cloud
    except Exception as exc:
        print(f"[WARN] HydroShear IsaacLab point-cloud sampler unavailable: {exc}", flush=True)
        return None

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        points = sample_object_point_cloud(1, candidate_count, str(prim_path), device="cpu")
        samples = points[0].detach().cpu().numpy().astype(np.float32)
    except Exception as exc:
        print(f"[WARN] HydroShear IsaacLab point-cloud sampling failed for {prim_path}: {exc}", flush=True)
        return None
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    samples = maybe_poisson_downsample_hydroshear_samples(
        samples,
        sample_mode=sample_mode,
        poisson_radius=poisson_radius,
        seed=seed,
        label="stage samples",
    )
    print(
        f"[INFO] HydroShear object mesh samples: {samples.shape[0]} points from stage prim {prim_path} "
        f"using IsaacLab sample_object_point_cloud ({sample_mode})",
        flush=True,
    )
    return samples


def stats_line(force: np.ndarray, tacmap: np.ndarray | None, labels: list[str]) -> str:
    parts = []
    force_np = _to_numpy_float32(force)
    for i in range(force_np.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = force_np[i]
        parts.append(f"force/{label}={_grid_active_summary(grid, threshold=float(args.active_threshold))}")
    if tacmap is not None:
        tacmap_np = _to_numpy_float32(tacmap)
        for i in range(tacmap_np.shape[0]):
            label = labels[i] if i < len(labels) else f"S{i}"
            grid = tacmap_np[i]
            active = int(np.count_nonzero(grid > 0))
            parts.append(f"tacmap/{label}=max:{int(grid.max())} active:{active}/{grid.size}")
    return ", ".join(parts)


def fots_stats_line(fots_output, labels: list[str]) -> str:
    if fots_output is None:
        return ""
    parts = []
    max_depths = _to_numpy_float32(fots_output.max_depth_mm).reshape(-1)
    active_pixels = _to_numpy_float32(fots_output.active_pixels).reshape(-1)
    marker_flow = _to_numpy_float32(fots_output.marker_flow)
    theta_rad = _to_numpy_float32(fots_output.theta_rad).reshape(-1) if getattr(fots_output, "theta_rad", None) is not None else None
    theta_delta_rad = (
        _to_numpy_float32(fots_output.theta_delta_rad).reshape(-1)
        if getattr(fots_output, "theta_delta_rad", None) is not None
        else None
    )
    for i, (max_depth, active) in enumerate(zip(max_depths, active_pixels)):
        label = labels[i] if i < len(labels) else f"S{i}"
        flow = marker_flow[i]
        disp = np.linalg.norm(flow[1] - flow[0], axis=-1)
        theta_deg = ""
        if theta_rad is not None and i < len(theta_rad):
            theta_deg = f" theta:{float(np.rad2deg(theta_rad[i])):.2f}deg"
        theta_delta_deg = ""
        if theta_delta_rad is not None and i < len(theta_delta_rad):
            theta_delta_deg = f" dtheta:{float(np.rad2deg(theta_delta_rad[i])):.2f}deg"
        parts.append(
            f"fots/{label}=depth:{float(max_depth):.3f}mm active:{int(active)} "
            f"marker_max:{float(disp.max()):.3f}px{theta_deg}{theta_delta_deg}"
        )
    return ", ".join(parts)


def tacex_rgb_stats_line(tacex_rgb_output, labels: list[str]) -> str:
    if tacex_rgb_output is None:
        return ""
    parts = []
    max_depths = _to_numpy_float32(tacex_rgb_output.max_depth_mm).reshape(-1)
    active_pixels = _to_numpy_float32(tacex_rgb_output.active_pixels).reshape(-1)
    for i, (max_depth, active) in enumerate(zip(max_depths, active_pixels)):
        label = labels[i] if i < len(labels) else f"S{i}"
        parts.append(f"tacex_rgb/{label}=depth:{float(max_depth):.3f}mm active:{int(active)}")
    return ", ".join(parts)


def tacsl_shear_stats_line(tacsl_shear_output, labels: list[str]) -> str:
    if tacsl_shear_output is None:
        return ""
    parts = []
    max_depth_mm = _to_numpy_float32(tacsl_shear_output.max_depth_mm).reshape(-1)
    active_taxels_arr = _to_numpy_float32(tacsl_shear_output.active_taxels).reshape(-1)
    max_normal_force = _to_numpy_float32(tacsl_shear_output.max_normal_force).reshape(-1)
    max_shear_force = _to_numpy_float32(tacsl_shear_output.max_shear_force).reshape(-1)
    for i, (max_depth, active_taxels, max_normal, max_shear) in enumerate(
        zip(
            max_depth_mm,
            active_taxels_arr,
            max_normal_force,
            max_shear_force,
        )
    ):
        label = labels[i] if i < len(labels) else f"S{i}"
        parts.append(
            f"tacsl_shear/{label}=depth:{float(max_depth):.3f}mm "
            f"active_taxels:{int(active_taxels)} "
            f"normal_max:{float(max_normal):.6f} shear_max:{float(max_shear):.6f}"
        )
    return ", ".join(parts)


def hydroshear_marker_stats_line(hydroshear_output, labels: list[str]) -> str:
    if hydroshear_output is None:
        return ""
    parts = []
    max_depth_mm = _to_numpy_float32(hydroshear_output.max_depth_mm).reshape(-1)
    active_markers_arr = _to_numpy_float32(hydroshear_output.active_markers).reshape(-1)
    active_objects_arr = _to_numpy_float32(hydroshear_output.active_object_samples).reshape(-1)
    max_marker_motion_px = _to_numpy_float32(hydroshear_output.max_marker_motion_px).reshape(-1)
    for i, (max_depth, active_markers, active_objects, marker_motion) in enumerate(
        zip(
            max_depth_mm,
            active_markers_arr,
            active_objects_arr,
            max_marker_motion_px,
        )
    ):
        label = labels[i] if i < len(labels) else f"S{i}"
        parts.append(
            f"hydroshear/{label}=depth:{float(max_depth):.3f}mm "
            f"active_markers:{int(active_markers)} object_samples:{int(active_objects)} "
            f"marker_max:{float(marker_motion):.3f}px"
        )
    return ", ".join(parts)


def tacmap_raw_stats_line(tacmap_raw: np.ndarray | None, labels: list[str]) -> str:
    if tacmap_raw is None:
        return ""
    parts = []
    for i in range(tacmap_raw.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = np.nan_to_num(np.asarray(tacmap_raw[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        active = int(np.count_nonzero(grid > 0.0))
        parts.append(f"tacmap_raw/{label}=max:{float(grid.max()) * 1000.0:.3f}mm active:{active}/{grid.size}")
    return ", ".join(parts)


def normal_ray_stats_line(normal_ray_pressure: np.ndarray | None, labels: list[str]) -> str:
    if normal_ray_pressure is None:
        return ""
    parts = []
    for i in range(normal_ray_pressure.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = np.nan_to_num(np.asarray(normal_ray_pressure[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        active = int(np.count_nonzero(grid > float(args.active_threshold)))
        parts.append(f"normal_ray/{label}=max:{float(grid.max()):.4f} active:{active}/{grid.size}")
    return ", ".join(parts)


def geometry_normal_ray_stats_line(pressure: np.ndarray | None, labels: list[str]) -> str:
    if pressure is None:
        return ""
    parts = []
    for i in range(pressure.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = np.nan_to_num(np.asarray(pressure[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        active = int(np.count_nonzero(grid > float(args.active_threshold)))
        parts.append(f"geometry_normal_ray/{label}=max:{float(grid.max()):.4f} active:{active}/{grid.size}")
    return ", ".join(parts)


def physx_contact_stats_line(
    physx_contact: np.ndarray | None,
    physx_contact_raw: np.ndarray | None,
    physx_contact_count: np.ndarray | None,
    labels: list[str],
) -> str:
    if physx_contact is None:
        return ""
    parts = []
    for i in range(physx_contact.shape[0]):
        label = labels[i] if i < len(labels) else f"S{i}"
        grid = np.nan_to_num(np.asarray(physx_contact[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        raw_grid = (
            np.nan_to_num(np.asarray(physx_contact_raw[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if physx_contact_raw is not None and i < physx_contact_raw.shape[0]
            else grid
        )
        contact_count = (
            int(float(physx_contact_count[i]))
            if physx_contact_count is not None and i < len(physx_contact_count)
            else 0
        )
        active = int(np.count_nonzero(grid > float(args.active_threshold)))
        parts.append(
            f"physx_contact/{label}=max:{float(grid.max()):.4f} "
            f"raw_max:{float(raw_grid.max()):.3f}N contacts:{contact_count} active:{active}/{grid.size}"
        )
    return ", ".join(parts)


def press_offset_for_step(step: int, cfg: IntegratedTactileEnvCfg) -> float:
    denom = max(1, int(cfg.press_steps) - 1)
    counter = min(step + 1, max(0, int(cfg.press_steps) - 1))
    alpha = min(max(float(counter) / float(denom), 0.0), 1.0)
    return float(cfg.press_start_offset) + alpha * (float(cfg.press_end_offset) - float(cfg.press_start_offset))


def grid_benchmark_metrics(
    name: str,
    grid: np.ndarray | None,
    *,
    threshold: float,
) -> dict[str, float | int | str]:
    if grid is None:
        return {
            f"{name}_max": 0.0,
            f"{name}_sum": 0.0,
            f"{name}_active": 0,
            f"{name}_centroid_row": "",
            f"{name}_centroid_col": "",
        }

    values = np.nan_to_num(np.asarray(grid, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    active_mask = values > float(threshold)
    positive = np.where(active_mask, values, 0.0)
    total = float(positive.sum())
    row_c = ""
    col_c = ""
    if total > 0.0:
        rows, cols = np.indices(values.shape)
        row_c = float((rows * positive).sum() / total)
        col_c = float((cols * positive).sum() / total)
    return {
        f"{name}_max": float(values.max()) if values.size else 0.0,
        f"{name}_sum": float(values.sum()) if values.size else 0.0,
        f"{name}_active": int(np.count_nonzero(active_mask)),
        f"{name}_centroid_row": row_c,
        f"{name}_centroid_col": col_c,
    }


def contact_benchmark_row(
    *,
    step: int,
    cfg: IntegratedTactileEnvCfg,
    force: np.ndarray,
    pressure_force_raw: np.ndarray | None,
    physx_contact_force: np.ndarray | None,
    physx_contact_force_raw: np.ndarray | None,
    physx_contact_count: np.ndarray | None,
    tacmap: np.ndarray | None,
    tacmap_raw: np.ndarray | None,
    sensor_index: int,
    force_threshold: float,
    depth_threshold_m: float,
) -> dict[str, float | int | str]:
    idx = max(0, int(sensor_index))

    def pick(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None or idx >= arr.shape[0]:
            return None
        return arr[idx]

    offset_m = press_offset_for_step(step, cfg) if bool(cfg.enable_press_motion) else 0.0
    row: dict[str, float | int | str] = {
        "step": int(step),
        "sensor_index": int(idx),
        "command_offset_m": float(offset_m),
        "command_offset_mm": float(offset_m * 1000.0),
        "physx_contact_count": (
            int(float(physx_contact_count[idx]))
            if physx_contact_count is not None and idx < len(physx_contact_count)
            else 0
        ),
    }
    row.update(grid_benchmark_metrics("warpsdf_force", pick(force), threshold=force_threshold))
    row.update(grid_benchmark_metrics("warpsdf_raw_force_n", pick(pressure_force_raw), threshold=0.0))
    row.update(grid_benchmark_metrics("physx_force", pick(physx_contact_force), threshold=force_threshold))
    row.update(grid_benchmark_metrics("physx_raw_force_n", pick(physx_contact_force_raw), threshold=0.0))
    row.update(grid_benchmark_metrics("tacmap_quantized", pick(tacmap), threshold=0.0))
    row.update(grid_benchmark_metrics("tacmap_depth_m", pick(tacmap_raw), threshold=depth_threshold_m))
    return row


def _first_onset(rows: list[dict[str, float | int | str]], key: str) -> dict[str, float | int | str] | None:
    for row in rows:
        if float(row.get(key, 0) or 0) > 0:
            return row
    return None


def contact_benchmark_summary(rows: list[dict[str, float | int | str]]) -> dict[str, object]:
    summary: dict[str, object] = {"num_steps": len(rows)}
    for name, key in (
        ("warpsdf", "warpsdf_force_active"),
        ("physx", "physx_contact_count"),
        ("physx_force_map", "physx_force_active"),
        ("tacmap", "tacmap_depth_m_active"),
    ):
        onset = _first_onset(rows, key)
        summary[f"{name}_onset_step"] = int(onset["step"]) if onset else None
        summary[f"{name}_onset_offset_mm"] = float(onset["command_offset_mm"]) if onset else None

    tacmap_active = [row for row in rows if int(row.get("tacmap_depth_m_active", 0) or 0) > 0]
    physx_active = [row for row in rows if int(row.get("physx_contact_count", 0) or 0) > 0]
    summary["physx_misses_when_tacmap_active"] = sum(
        1 for row in tacmap_active if int(row.get("physx_contact_count", 0) or 0) <= 0
    )
    summary["physx_precontact_hits_before_tacmap"] = sum(
        1
        for row in physx_active
        if int(row.get("tacmap_depth_m_active", 0) or 0) <= 0
    )
    summary["physx_contact_count_nonzero_steps"] = len(physx_active)
    return summary


def write_contact_benchmark(rows: list[dict[str, float | int | str]], out_arg: str | None) -> None:
    if not out_arg:
        return
    out_path = Path(out_arg).expanduser()
    if out_path.suffix.lower() not in {".csv", ".json"}:
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "contact_benchmark.csv"
        json_path = out_path / "contact_benchmark_summary.json"
    elif out_path.suffix.lower() == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = out_path
        csv_path = out_path.with_suffix(".csv")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = out_path
        json_path = out_path.with_suffix(".summary.json")

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = contact_benchmark_summary(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[BENCHMARK] contact trace: {csv_path}", flush=True)
    print(f"[BENCHMARK] contact summary: {json_path}", flush=True)
    print(f"[BENCHMARK] summary: {json.dumps(summary, sort_keys=True)}", flush=True)


def link_surface_raw_stats_line(
    surface_raw: np.ndarray | None,
    object_raw: np.ndarray | None,
    labels: list[str],
) -> str:
    if surface_raw is None or object_raw is None:
        return ""
    count = min(int(surface_raw.shape[0]), int(object_raw.shape[0]))
    parts = []
    for i in range(count):
        label = labels[i] if i < len(labels) else f"S{i}"
        surface_grid = np.nan_to_num(
            np.asarray(surface_raw[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        object_grid = np.nan_to_num(
            np.asarray(object_raw[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        surface_active = int(np.count_nonzero(surface_grid > 0.0))
        object_active = int(np.count_nonzero(object_grid > 0.0))
        paired = (surface_grid > 0.0) & (object_grid > 0.0)
        before_surface = int(np.count_nonzero(paired & (object_grid <= surface_grid)))
        if np.any(paired):
            gap_mm = float(np.min(object_grid[paired] - surface_grid[paired]) * 1000.0)
            pen_mm = float(np.max(np.maximum(surface_grid[paired] - object_grid[paired], 0.0)) * 1000.0)
            gap_text = f" gap_min:{gap_mm:.3f}mm pen_max:{pen_mm:.3f}mm"
        else:
            gap_text = ""
        parts.append(
            f"link_surface/{label}=surface:{surface_active}/{surface_grid.size} "
            f"object:{object_active}/{object_grid.size} before_surface:{before_surface}{gap_text}"
        )
    return ", ".join(parts)


def _rgb_to_bmp_bytes(img_rgb: np.ndarray) -> bytes:
    img = to_numpy_uint8_rgb(img_rgb)
    h, w, _ = img.shape
    row_stride = ((w * 3 + 3) // 4) * 4
    padding = row_stride - w * 3
    pixel_rows = [row[:, ::-1].tobytes() + b"\x00" * padding for row in img[::-1]]
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
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def do_GET(self):
            if self.path.startswith("/map.bmp"):
                with _LIVE_LOCK:
                    img = placeholder if _LIVE_IMAGE_RGB is None else _LIVE_IMAGE_RGB.copy()
                self._send("image/bmp", _rgb_to_bmp_bytes(img))
                return
            if self.path.startswith("/stats"):
                with _LIVE_LOCK:
                    stats = _LIVE_STATS
                self._send("text/plain; charset=utf-8", stats.encode("utf-8"))
                return
            html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Integrated Tactile</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}main{padding:16px}
img{image-rendering:pixelated;max-width:100%;border:1px solid #444}pre{font-size:16px}</style></head>
<body><main><pre id="stats">loading...</pre><img id="map" src="/map.bmp" /></main>
<script>
async function refresh(){
  document.getElementById("map").src="/map.bmp?t="+Date.now();
  document.getElementById("stats").textContent=await fetch("/stats?t="+Date.now()).then(r=>r.text());
}
setInterval(refresh, 100); refresh();
</script></body></html>"""
            self._send("text/html; charset=utf-8", html.encode("utf-8"))

    server = ThreadingHTTPServer(("0.0.0.0", int(port)), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[LIVE] Open http://localhost:{port} to view integrated tactile maps.", flush=True)
    return server


def update_live_web(img_rgb, stats: str):
    if not args.live_web:
        return
    global _LIVE_IMAGE_RGB, _LIVE_STATS
    img_np = to_numpy_uint8_rgb(img_rgb)
    with _LIVE_LOCK:
        _LIVE_IMAGE_RGB = img_np.copy()
        _LIVE_STATS = stats


class _IntegratedPanel:
    def __init__(self, height: int, width: int):
        import omni.ui as ui

        self._height = int(height)
        self._width = int(width)
        self._win = ui.Window("Integrated Tactile", width=self._width + 24, height=self._height + 76)
        with self._win.frame:
            with ui.VStack(spacing=6):
                self._stats = ui.Label("waiting")
                self._provider = ui.ByteImageProvider()
                ui.ImageWithProvider(self._provider, width=self._width, height=self._height)

    def update(self, img_rgb, stats: str):
        rgb = to_numpy_uint8_rgb(img_rgb)
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1), dtype=np.uint8)
        self._provider.set_bytes_data(memoryview(rgba.reshape(-1)), (self._width, self._height))
        self._stats.text = stats


def maybe_show_cv(img_rgb):
    if not args.show_cv:
        return
    try:
        import cv2
        img_np = to_numpy_uint8_rgb(img_rgb)
        cv2.imshow("Integrated Tactile", img_np[:, :, ::-1])
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[WARN] OpenCV display disabled: {exc}", flush=True)
        args.show_cv = False


def maybe_save(img_rgb, step: int):
    if not args.save_maps or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_np = to_numpy_uint8_rgb(img_rgb)
    try:
        from PIL import Image
        Image.fromarray(img_np).save(out_dir / f"integrated_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"integrated_{step:06d}.npy", img_np)


def maybe_save_heightmap(tacmap_raw: np.ndarray, step: int):
    if not args.save_heightmaps or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "heightmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"tacmap_raw_{step:06d}.npy", _to_numpy_float32(tacmap_raw))


def maybe_save_fots(fots_output, step: int):
    if fots_output is None or not args.save_fots or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "fots"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"fots_marker_flow_{step:06d}.npy", _to_numpy_float32(fots_output.marker_flow))
    try:
        from PIL import Image

        Image.fromarray(to_numpy_uint8_rgb(image_strip(fots_output.marker_images))).save(out_dir / f"fots_marker_flow_{step:06d}.png")
        Image.fromarray(to_numpy_uint8_rgb(image_strip(fots_output.marker_overlay_images))).save(out_dir / f"fots_marker_overlay_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"fots_marker_flow_img_{step:06d}.npy", to_numpy_uint8_rgb(fots_output.marker_images))
        np.save(out_dir / f"fots_marker_overlay_{step:06d}.npy", to_numpy_uint8_rgb(fots_output.marker_overlay_images))


def maybe_save_tacex_rgb(tacex_rgb_output, step: int):
    if tacex_rgb_output is None or not args.save_tacex_rgb or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "tacex_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"tacex_height_map_mm_{step:06d}.npy", _to_numpy_float32(tacex_rgb_output.taxim_height_map_mm))
    try:
        from PIL import Image

        Image.fromarray(to_numpy_uint8_rgb(image_strip(tacex_rgb_output.tactile_rgb))).save(out_dir / f"tacex_rgb_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"tacex_rgb_{step:06d}.npy", to_numpy_uint8_rgb(tacex_rgb_output.tactile_rgb))


def load_pressure_calibration(
    path_arg: str | None,
    *,
    image_shape: tuple[int, int],
) -> dict[str, object]:
    return load_pressure_calibration_overrides(path_arg, image_shape=image_shape)


def apply_pressure_calibration_overrides(cfg: IntegratedTactileEnvCfg, overrides: dict[str, object]) -> None:
    if "stiffness" in overrides:
        cfg.stiffness = overrides["stiffness"]
    if "damping" in overrides:
        cfg.damping = overrides["damping"]
    if "max_force" in overrides:
        cfg.max_force = overrides["max_force"]
    if "gain" in overrides:
        cfg.pressure_gain = overrides["gain"]
    if "bias" in overrides:
        cfg.pressure_bias = overrides["bias"]
    if "gamma" in overrides:
        cfg.pressure_gamma = overrides["gamma"]
    if "threshold" in overrides:
        cfg.pressure_threshold = overrides["threshold"]
    if "taxel_area" in overrides:
        cfg.taxel_area = overrides["taxel_area"]


def cli_arg_provided(option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])


def calibration_value_metadata(value) -> float | dict[str, object]:
    return pressure_calibration_value_summary(value)


def calibration_value_text(value) -> str:
    summary = pressure_calibration_value_summary(value)
    if isinstance(summary, dict):
        return (
            f"array{tuple(summary['shape'])}"
            f"[{float(summary['min']):.4g},{float(summary['max']):.4g}]"
        )
    return f"{float(summary):g}"


class PressureTraceRecorder:
    def __init__(self, out_dir: str | Path, metadata: dict[str, object], layout_arrays: dict[str, np.ndarray] | None = None):
        self.out_dir = Path(out_dir).expanduser()
        self.run_id = str(metadata.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.metadata = dict(metadata)
        self.metadata["run_id"] = self.run_id
        self.layout_arrays = {str(k): np.asarray(v) for k, v in (layout_arrays or {}).items()}
        self.steps: list[int] = []
        self.pressure_penetration_m: list[np.ndarray] = []
        self.pressure_signed_distance_m: list[np.ndarray] = []
        self.pressure_penetration_velocity_mps: list[np.ndarray] = []
        self.pressure_raw_n: list[np.ndarray] = []
        self.pressure_norm: list[np.ndarray] = []
        self.total_force_n: list[np.ndarray] = []
        self.center_of_pressure_px: list[np.ndarray] = []
        self.tacmap_raw_m: list[np.ndarray] = []
        self.physx_contact_count: list[np.ndarray] = []
        self.normal_ray_penetration_m: list[np.ndarray] = []
        self.normal_ray_signed_distance_m: list[np.ndarray] = []
        self.normal_ray_penetration_velocity_mps: list[np.ndarray] = []
        self.normal_ray_pressure_raw_n: list[np.ndarray] = []
        self.normal_ray_pressure_norm: list[np.ndarray] = []
        self.normal_ray_total_force_n: list[np.ndarray] = []
        self.normal_ray_center_of_pressure_px: list[np.ndarray] = []
        self.geometry_normal_ray_penetration_m: list[np.ndarray] = []
        self.geometry_normal_ray_signed_distance_m: list[np.ndarray] = []
        self.geometry_normal_ray_penetration_velocity_mps: list[np.ndarray] = []
        self.geometry_normal_ray_pressure_raw_n: list[np.ndarray] = []
        self.geometry_normal_ray_pressure_norm: list[np.ndarray] = []
        self.geometry_normal_ray_total_force_n: list[np.ndarray] = []
        self.geometry_normal_ray_center_of_pressure_px: list[np.ndarray] = []
        self.geometry_normal_ray_sample_support_fraction: list[np.ndarray] = []
        self.geometry_normal_ray_sample_active_count: list[np.ndarray] = []
        self.geometry_normal_ray_sample_mean_penetration_m: list[np.ndarray] = []
        self.geometry_normal_ray_sample_positive_mean_penetration_m: list[np.ndarray] = []
        self.geometry_normal_ray_sample_max_penetration_m: list[np.ndarray] = []
        self.geometry_normal_ray_sample_penetrations_m: list[np.ndarray] = []
        self.geometry_normal_ray_sample_offsets_l: list[np.ndarray] = []
        self.geometry_normal_ray_sample_points_l_m: list[np.ndarray] = []

    def record(
        self,
        *,
        step: int,
        pressure_penetration_m: np.ndarray | None,
        pressure_signed_distance_m: np.ndarray | None,
        pressure_penetration_velocity_mps: np.ndarray | None,
        pressure_raw_n: np.ndarray | None,
        pressure_norm: np.ndarray | None,
        tacmap_raw_m: np.ndarray | None,
        physx_contact_count: np.ndarray | None,
        normal_ray_penetration_m: np.ndarray | None = None,
        normal_ray_signed_distance_m: np.ndarray | None = None,
        normal_ray_penetration_velocity_mps: np.ndarray | None = None,
        normal_ray_pressure_raw_n: np.ndarray | None = None,
        normal_ray_pressure_norm: np.ndarray | None = None,
        normal_ray_total_force_n: np.ndarray | None = None,
        normal_ray_center_of_pressure_px: np.ndarray | None = None,
        geometry_normal_ray_penetration_m: np.ndarray | None = None,
        geometry_normal_ray_signed_distance_m: np.ndarray | None = None,
        geometry_normal_ray_penetration_velocity_mps: np.ndarray | None = None,
        geometry_normal_ray_pressure_raw_n: np.ndarray | None = None,
        geometry_normal_ray_pressure_norm: np.ndarray | None = None,
        geometry_normal_ray_total_force_n: np.ndarray | None = None,
        geometry_normal_ray_center_of_pressure_px: np.ndarray | None = None,
        geometry_normal_ray_sample_support_fraction: np.ndarray | None = None,
        geometry_normal_ray_sample_active_count: np.ndarray | None = None,
        geometry_normal_ray_sample_mean_penetration_m: np.ndarray | None = None,
        geometry_normal_ray_sample_positive_mean_penetration_m: np.ndarray | None = None,
        geometry_normal_ray_sample_max_penetration_m: np.ndarray | None = None,
        geometry_normal_ray_sample_penetrations_m: np.ndarray | None = None,
        geometry_normal_ray_sample_offsets_l: np.ndarray | None = None,
        geometry_normal_ray_sample_points_l_m: np.ndarray | None = None,
    ) -> None:
        self.steps.append(int(step))
        pressure_raw_arr = _trace_array(pressure_raw_n)
        self.pressure_penetration_m.append(_trace_array(pressure_penetration_m))
        self.pressure_signed_distance_m.append(_trace_array(pressure_signed_distance_m))
        self.pressure_penetration_velocity_mps.append(_trace_array(pressure_penetration_velocity_mps))
        self.pressure_raw_n.append(pressure_raw_arr)
        self.pressure_norm.append(_trace_array(pressure_norm))
        if pressure_raw_arr.ndim >= 2 and pressure_raw_arr.shape[-2] > 0 and pressure_raw_arr.shape[-1] > 0:
            total, center = pressure_map_stats(pressure_raw_arr)
        else:
            total = np.zeros((0,), dtype=np.float32)
            center = np.zeros((0, 2), dtype=np.float32)
        self.total_force_n.append(total)
        self.center_of_pressure_px.append(center)
        self.tacmap_raw_m.append(_trace_array(tacmap_raw_m))
        self.physx_contact_count.append(_trace_array(physx_contact_count))
        self.normal_ray_penetration_m.append(_trace_array(normal_ray_penetration_m))
        self.normal_ray_signed_distance_m.append(_trace_array(normal_ray_signed_distance_m))
        self.normal_ray_penetration_velocity_mps.append(_trace_array(normal_ray_penetration_velocity_mps))
        self.normal_ray_pressure_raw_n.append(_trace_array(normal_ray_pressure_raw_n))
        self.normal_ray_pressure_norm.append(_trace_array(normal_ray_pressure_norm))
        self.normal_ray_total_force_n.append(_trace_array(normal_ray_total_force_n))
        self.normal_ray_center_of_pressure_px.append(_trace_array(normal_ray_center_of_pressure_px))
        self.geometry_normal_ray_penetration_m.append(_trace_array(geometry_normal_ray_penetration_m))
        self.geometry_normal_ray_signed_distance_m.append(_trace_array(geometry_normal_ray_signed_distance_m))
        self.geometry_normal_ray_penetration_velocity_mps.append(_trace_array(geometry_normal_ray_penetration_velocity_mps))
        self.geometry_normal_ray_pressure_raw_n.append(_trace_array(geometry_normal_ray_pressure_raw_n))
        self.geometry_normal_ray_pressure_norm.append(_trace_array(geometry_normal_ray_pressure_norm))
        self.geometry_normal_ray_total_force_n.append(_trace_array(geometry_normal_ray_total_force_n))
        self.geometry_normal_ray_center_of_pressure_px.append(_trace_array(geometry_normal_ray_center_of_pressure_px))
        self.geometry_normal_ray_sample_support_fraction.append(
            _trace_array(geometry_normal_ray_sample_support_fraction)
        )
        self.geometry_normal_ray_sample_active_count.append(_trace_array(geometry_normal_ray_sample_active_count))
        self.geometry_normal_ray_sample_mean_penetration_m.append(
            _trace_array(geometry_normal_ray_sample_mean_penetration_m)
        )
        self.geometry_normal_ray_sample_positive_mean_penetration_m.append(
            _trace_array(geometry_normal_ray_sample_positive_mean_penetration_m)
        )
        self.geometry_normal_ray_sample_max_penetration_m.append(
            _trace_array(geometry_normal_ray_sample_max_penetration_m)
        )
        self.geometry_normal_ray_sample_penetrations_m.append(
            _trace_array(geometry_normal_ray_sample_penetrations_m)
        )
        self.geometry_normal_ray_sample_offsets_l.append(_trace_array(geometry_normal_ray_sample_offsets_l))
        self.geometry_normal_ray_sample_points_l_m.append(_trace_array(geometry_normal_ray_sample_points_l_m))

    def close(self) -> tuple[Path, Path] | None:
        if not self.steps:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = self.out_dir / f"{self.run_id}.npz"
        metadata_path = self.out_dir / f"{self.run_id}.metadata.json"
        metadata_json = json.dumps(self.metadata, sort_keys=True)
        payload = dict(
            step=np.asarray(self.steps, dtype=np.int64),
            penetration_m=_stack_trace(self.pressure_penetration_m),
            signed_distance_m=_stack_trace(self.pressure_signed_distance_m),
            penetration_velocity_mps=_stack_trace(self.pressure_penetration_velocity_mps),
            pressure_raw_n=_stack_trace(self.pressure_raw_n),
            pressure_norm=_stack_trace(self.pressure_norm),
            total_force_n=_stack_trace(self.total_force_n),
            center_of_pressure_px=_stack_trace(self.center_of_pressure_px),
            tacmap_raw_m=_stack_trace(self.tacmap_raw_m),
            physx_contact_count=_stack_trace(self.physx_contact_count),
            normal_ray_penetration_m=_stack_trace(self.normal_ray_penetration_m),
            normal_ray_signed_distance_m=_stack_trace(self.normal_ray_signed_distance_m),
            normal_ray_penetration_velocity_mps=_stack_trace(self.normal_ray_penetration_velocity_mps),
            normal_ray_pressure_raw_n=_stack_trace(self.normal_ray_pressure_raw_n),
            normal_ray_pressure_norm=_stack_trace(self.normal_ray_pressure_norm),
            normal_ray_total_force_n=_stack_trace(self.normal_ray_total_force_n),
            normal_ray_center_of_pressure_px=_stack_trace(self.normal_ray_center_of_pressure_px),
            geometry_normal_ray_penetration_m=_stack_trace(self.geometry_normal_ray_penetration_m),
            geometry_normal_ray_signed_distance_m=_stack_trace(self.geometry_normal_ray_signed_distance_m),
            geometry_normal_ray_penetration_velocity_mps=_stack_trace(self.geometry_normal_ray_penetration_velocity_mps),
            geometry_normal_ray_pressure_raw_n=_stack_trace(self.geometry_normal_ray_pressure_raw_n),
            geometry_normal_ray_pressure_norm=_stack_trace(self.geometry_normal_ray_pressure_norm),
            geometry_normal_ray_total_force_n=_stack_trace(self.geometry_normal_ray_total_force_n),
            geometry_normal_ray_center_of_pressure_px=_stack_trace(self.geometry_normal_ray_center_of_pressure_px),
            geometry_normal_ray_sample_support_fraction=_stack_trace(
                self.geometry_normal_ray_sample_support_fraction
            ),
            geometry_normal_ray_sample_active_count=_stack_trace(self.geometry_normal_ray_sample_active_count),
            geometry_normal_ray_sample_mean_penetration_m=_stack_trace(
                self.geometry_normal_ray_sample_mean_penetration_m
            ),
            geometry_normal_ray_sample_positive_mean_penetration_m=_stack_trace(
                self.geometry_normal_ray_sample_positive_mean_penetration_m
            ),
            geometry_normal_ray_sample_max_penetration_m=_stack_trace(
                self.geometry_normal_ray_sample_max_penetration_m
            ),
            geometry_normal_ray_sample_penetrations_m=_stack_trace(
                self.geometry_normal_ray_sample_penetrations_m
            ),
            geometry_normal_ray_sample_offsets_l=_stack_trace(self.geometry_normal_ray_sample_offsets_l),
            geometry_normal_ray_sample_points_l_m=_stack_trace(self.geometry_normal_ray_sample_points_l_m),
            metadata_json=np.asarray(metadata_json),
        )
        payload.update(self.layout_arrays)
        np.savez_compressed(npz_path, **payload)
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")
        return npz_path, metadata_path


def _trace_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    return np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _stack_trace(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0,), dtype=np.float32)
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) == 1:
        return np.stack(values, axis=0)
    return np.asarray(values, dtype=object)


def pressure_verify_output_path(npz_path: Path, out_arg: str | None) -> Path:
    if not out_arg:
        return npz_path.with_suffix(".verify.json")
    out = Path(out_arg).expanduser()
    if out.suffix.lower() == ".json":
        return out
    return out / f"{npz_path.stem}.verify.json"


def write_pressure_trace_verification(npz_path: Path) -> bool:
    out_path = pressure_verify_output_path(npz_path, args.pressure_verify_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = pressure_trace_report(
            npz_path,
            pressure_key=str(args.pressure_verify_pressure_key),
            raw_pressure_key=str(args.pressure_verify_raw_key),
            penetration_key=str(args.pressure_verify_penetration_key),
            reference_key=args.pressure_verify_reference_key,
            reference_valid_mask_key=args.pressure_verify_reference_valid_mask_key,
            reference_layer=args.pressure_verify_reference_layer,
            active_threshold=float(args.pressure_verify_active_threshold),
            penetration_threshold=float(args.pressure_verify_penetration_threshold),
            reference_threshold=float(args.pressure_verify_reference_threshold),
        )
        evaluation = evaluate_pressure_trace_report(
            report,
            precontact_leakage_threshold=float(args.pressure_verify_precontact_leakage_threshold),
            reference_iou_threshold=float(args.pressure_verify_reference_iou_threshold),
            reference_alignment_valid_fraction_threshold=args.pressure_verify_reference_alignment_valid_fraction_threshold,
            reference_contact_valid_fraction_threshold=args.pressure_verify_reference_contact_valid_fraction_threshold,
            reference_invalid_contact_fraction_threshold=args.pressure_verify_reference_invalid_contact_fraction_threshold,
            centroid_error_threshold_px=float(args.pressure_verify_centroid_threshold_px),
            bbox_error_threshold_px=float(args.pressure_verify_bbox_threshold_px),
            depth_rmse_threshold_m=float(args.pressure_verify_depth_rmse_threshold_m),
            onset_error_threshold_frames=int(args.pressure_verify_onset_threshold_frames),
            offset_error_threshold_frames=int(args.pressure_verify_offset_threshold_frames),
            force_depth_spearman_threshold=float(args.pressure_verify_spearman_threshold),
            dense_reference_layers=(
                tuple(str(value) for value in args.pressure_verify_dense_reference_layer)
                if args.pressure_verify_dense_reference_layer is not None
                else DEFAULT_DENSE_REFERENCE_ACCEPTANCE_LAYERS
            ),
            allow_reference_layer_override=bool(args.pressure_verify_allow_reference_layer_override),
        )
        contract = validate_pressure_trace_v1(npz_path)
        payload = {"contract": contract, "report": report, "evaluation": evaluation}
        passed = bool(evaluation.get("passed", False)) and bool(contract.get("passed", False))
    except Exception as exc:
        payload = {"error": str(exc), "evaluation": {"passed": False, "checks": []}}
        passed = False
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[VERIFY] pressure trace verification: {out_path}", flush=True)
    print(f"[VERIFY] passed={passed}", flush=True)
    return passed

def maybe_save_tacsl_shear(tacsl_shear_output, step: int):
    if tacsl_shear_output is None or not args.save_tacsl_shear or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "tacsl_shear"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"tacsl_penetration_depth_m_{step:06d}.npy", _to_numpy_float32(tacsl_shear_output.penetration_depth_m))
    np.save(out_dir / f"tacsl_normal_force_{step:06d}.npy", _to_numpy_float32(tacsl_shear_output.normal_force))
    np.save(out_dir / f"tacsl_shear_force_{step:06d}.npy", _to_numpy_float32(tacsl_shear_output.shear_force))
    try:
        from PIL import Image

        Image.fromarray(to_numpy_uint8_rgb(image_strip(tacsl_shear_output.shear_images))).save(out_dir / f"tacsl_shear_{step:06d}.png")
    except ImportError:
        np.save(out_dir / f"tacsl_shear_{step:06d}.npy", to_numpy_uint8_rgb(tacsl_shear_output.shear_images))


def maybe_save_hydroshear_marker(hydroshear_output, step: int):
    if hydroshear_output is None or not args.save_hydroshear_marker or step % max(1, int(args.save_every)) != 0:
        return
    out_dir = INTEGRATE_ROOT / "output" / "hydroshear_marker"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"hydroshear_marker_flow_{step:06d}.npy", _to_numpy_float32(hydroshear_output.marker_flow))
    np.save(
        out_dir / f"hydroshear_marker_displacement_m_{step:06d}.npy",
        _to_numpy_float32(hydroshear_output.displacement_m),
    )
    try:
        from PIL import Image

        Image.fromarray(to_numpy_uint8_rgb(image_strip(hydroshear_output.marker_images))).save(
            out_dir / f"hydroshear_marker_{step:06d}.png"
        )
        Image.fromarray(to_numpy_uint8_rgb(image_strip(hydroshear_output.debug_marker_displacement_images))).save(
            out_dir / f"hydroshear_debug_marker_displacement_{step:06d}.png"
        )
        Image.fromarray(to_numpy_uint8_rgb(image_strip(hydroshear_output.debug_sdf_images))).save(
            out_dir / f"hydroshear_debug_sdf_{step:06d}.png"
        )
        Image.fromarray(to_numpy_uint8_rgb(image_strip(hydroshear_output.debug_projection_images))).save(
            out_dir / f"hydroshear_debug_projection_{step:06d}.png"
        )
        Image.fromarray(to_numpy_uint8_rgb(image_strip(hydroshear_output.debug_mdilate_images))).save(
            out_dir / f"hydroshear_debug_mdilate_{step:06d}.png"
        )
        debug_grid = hydroshear_debug_image(hydroshear_output)
        if debug_grid is not None:
            Image.fromarray(to_numpy_uint8_rgb(debug_grid)).save(out_dir / f"hydroshear_debug_grid_{step:06d}.png")
    except ImportError:
        np.save(
            out_dir / f"hydroshear_marker_{step:06d}.npy",
            to_numpy_uint8_rgb(hydroshear_output.marker_images),
        )
        np.save(
            out_dir / f"hydroshear_debug_marker_displacement_{step:06d}.npy",
            to_numpy_uint8_rgb(hydroshear_output.debug_marker_displacement_images),
        )
        np.save(
            out_dir / f"hydroshear_debug_sdf_{step:06d}.npy",
            to_numpy_uint8_rgb(hydroshear_output.debug_sdf_images),
        )
        np.save(
            out_dir / f"hydroshear_debug_projection_{step:06d}.npy",
            to_numpy_uint8_rgb(hydroshear_output.debug_projection_images),
        )
        np.save(
            out_dir / f"hydroshear_debug_mdilate_{step:06d}.npy",
            to_numpy_uint8_rgb(hydroshear_output.debug_mdilate_images),
        )


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


def axis_to_vector(
    axis: str,
    *,
    grid_u_axis: str = "+y",
    grid_v_axis: str = "+z",
    ray_axis: str = "+x",
    ray_direction: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    axis = str(axis).strip().lower()
    sign = -1.0 if axis.startswith("-") else 1.0
    name = axis[1:] if axis.startswith(("+", "-")) else axis

    def xyz_axis(name_: str) -> np.ndarray:
        sign_ = -1.0 if str(name_).startswith("-") else 1.0
        key = str(name_)[-1].lower()
        return np.asarray(
            {
                "x": (sign_, 0.0, 0.0),
                "y": (0.0, sign_, 0.0),
                "z": (0.0, 0.0, sign_),
            }[key],
            dtype=np.float64,
        )

    if name == "u":
        vec = xyz_axis(grid_u_axis)
    elif name == "v":
        vec = xyz_axis(grid_v_axis)
    elif name == "ray":
        if ray_direction is None:
            vec = xyz_axis(ray_axis)
        else:
            vec = np.asarray(tuple(float(v) for v in ray_direction), dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            if norm < 1.0e-12:
                vec = xyz_axis(ray_axis)
            else:
                vec = vec / norm
    else:
        vec = xyz_axis(axis)
        sign = 1.0
    vec = sign * vec
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def axis_to_vector_np(axis: str) -> np.ndarray:
    return np.asarray(axis_to_vector(axis), dtype=np.float32)


def _normalized_vector_np(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _orthonormal_grid_axes_np(
    ray_axis: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ray_axis = _normalized_vector_np(ray_axis, axis_to_vector_np("+x"))

    u_axis = np.asarray(u_axis, dtype=np.float32) - float(np.dot(u_axis, ray_axis)) * ray_axis
    u_norm = float(np.linalg.norm(u_axis))
    if u_norm < 1.0e-8:
        for candidate in (
            np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        ):
            projected = candidate - float(np.dot(candidate, ray_axis)) * ray_axis
            candidate_norm = float(np.linalg.norm(projected))
            if candidate_norm >= 1.0e-8:
                u_axis = projected
                u_norm = candidate_norm
                break
    u_axis = u_axis / max(u_norm, 1.0e-8)

    v_axis = (
        np.asarray(v_axis, dtype=np.float32)
        - float(np.dot(v_axis, ray_axis)) * ray_axis
        - float(np.dot(v_axis, u_axis)) * u_axis
    )
    v_norm = float(np.linalg.norm(v_axis))
    if v_norm < 1.0e-8:
        v_axis = np.cross(ray_axis, u_axis)
        v_norm = float(np.linalg.norm(v_axis))
    v_axis = v_axis / max(v_norm, 1.0e-8)
    return u_axis.astype(np.float32), v_axis.astype(np.float32)


def _to_numpy_float32(value) -> np.ndarray:
    return _tensor_to_numpy_float32(value)


def _normalize_axis_np(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _grid_axes_from_points_and_dirs_np(points: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    ray_axis = _normalize_axis_np(np.mean(dirs.reshape(-1, 3), axis=0), axis_to_vector_np("+x"))
    if points.shape[1] > 1:
        u_vec = points[0, -1] - points[0, 0]
    else:
        u_vec = axis_to_vector_np("+y")
    if points.shape[0] > 1:
        v_vec = points[-1, 0] - points[0, 0]
    else:
        v_vec = axis_to_vector_np("+z")
    u_axis = _normalize_axis_np(u_vec, axis_to_vector_np("+y"))
    v_axis = _normalize_axis_np(v_vec, axis_to_vector_np("+z"))
    return np.stack([u_axis, v_axis, ray_axis], axis=0).astype(np.float32)


def _tacmap_sensor_local_grid(sensor, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    starts = getattr(sensor, "ray_starts_att", None)
    directions = getattr(sensor, "ray_directions_att", None)
    if starts is None or directions is None:
        return None
    starts_np = _to_numpy_float32(starts)
    dirs_np = _to_numpy_float32(directions)
    if starts_np.ndim != 3 or dirs_np.ndim != 3 or starts_np.shape[-1] != 3 or dirs_np.shape[-1] != 3:
        return None
    if starts_np.shape[1] != rows * cols or dirs_np.shape[1] != rows * cols:
        return None
    points = starts_np[0].reshape(rows, cols, 3).astype(np.float32)
    dirs = dirs_np[0].reshape(rows, cols, 3).astype(np.float32)
    axes = _grid_axes_from_points_and_dirs_np(points, dirs)
    return points, dirs, axes


def pressure_trace_layout_arrays(env: IntegratedTactileEnv, cfg: IntegratedTactileEnvCfg) -> dict[str, np.ndarray]:
    """Collect static pressure/TacMap local grids for offline reference alignment."""

    sensor_slots = list(getattr(env, "_sensor_slot_order", []) or [])
    tacmap_slots = list(getattr(env, "_tacmap_slot_order", []) or [])
    sensor_count = max(
        int(getattr(env, "_tactile_sensor_count", 0) or 0),
        max(sensor_slots) + 1 if sensor_slots else 0,
        max(tacmap_slots) + 1 if tacmap_slots else 0,
    )
    if sensor_count <= 0:
        return {}

    rows = max(1, int(cfg.num_rows))
    cols = max(1, int(cfg.num_cols))
    pressure_points = np.zeros((sensor_count, rows, cols, 3), dtype=np.float32)
    pressure_normals = np.zeros_like(pressure_points)
    pressure_valid = np.zeros((sensor_count,), dtype=np.uint8)
    pressure_link_names = np.asarray([""] * sensor_count, dtype="<U256")

    for sensor_index, sensor in enumerate(getattr(env, "_tactile_sensors", []) or []):
        slot = sensor_slots[sensor_index] if sensor_index < len(sensor_slots) else sensor_index
        if slot < 0 or slot >= sensor_count:
            continue
        maps = list(getattr(sensor, "pressure_taxel_maps", ()) or ())
        if not maps:
            continue
        taxel_map = maps[0]
        map_rows, map_cols = (int(v) for v in taxel_map.image_shape)
        if map_rows != rows or map_cols != cols:
            continue
        pressure_points[slot] = _to_numpy_float32(taxel_map.points_l).reshape(rows, cols, 3)
        pressure_normals[slot] = _to_numpy_float32(taxel_map.normals_l).reshape(rows, cols, 3)
        pressure_valid[slot] = 1
        pressure_link_names[slot] = str(taxel_map.link_name)

    layout_arrays: dict[str, np.ndarray] = {
        "pressure_taxel_points_l_m": pressure_points,
        "pressure_taxel_normals_l": pressure_normals,
        "pressure_taxel_layout_valid": pressure_valid,
        "pressure_taxel_link_name": pressure_link_names,
    }

    if str(getattr(cfg, "tacmap_ray_mode", "surface_normal")) != "link_surface":
        return layout_arrays

    tac_rows = max(1, int(getattr(env, "_tacmap_rows", cfg.tacmap_link_surface_height)))
    tac_cols = max(1, int(getattr(env, "_tacmap_cols", cfg.tacmap_link_surface_width)))
    tac_points = np.zeros((sensor_count, tac_rows, tac_cols, 3), dtype=np.float32)
    tac_dirs = np.zeros_like(tac_points)
    tac_axes = np.zeros((sensor_count, 3, 3), dtype=np.float32)
    tac_valid = np.zeros((sensor_count,), dtype=np.uint8)

    ray_axis = (
        axis_to_vector_np(str(cfg.tacmap_link_surface_ray_axis))
        if getattr(cfg, "tacmap_link_surface_ray_direction", None) is None
        else _normalized_vector_np(
            np.asarray(cfg.tacmap_link_surface_ray_direction, dtype=np.float32),
            axis_to_vector_np(str(cfg.tacmap_link_surface_ray_axis)),
        )
    )
    u_axis = axis_to_vector_np(str(cfg.tacmap_link_surface_grid_u_axis))
    v_axis = axis_to_vector_np(str(cfg.tacmap_link_surface_grid_v_axis))
    u_axis, v_axis = _orthonormal_grid_axes_np(ray_axis, u_axis, v_axis)
    center = np.asarray(cfg.tacmap_link_surface_grid_center, dtype=np.float32)
    u_values = np.linspace(
        -0.5 * float(cfg.tacmap_link_surface_grid_u_size),
        0.5 * float(cfg.tacmap_link_surface_grid_u_size),
        tac_cols,
        dtype=np.float32,
    )
    v_values = np.linspace(
        -0.5 * float(cfg.tacmap_link_surface_grid_v_size),
        0.5 * float(cfg.tacmap_link_surface_grid_v_size),
        tac_rows,
        dtype=np.float32,
    )
    grid_u, grid_v = np.meshgrid(u_values, v_values, indexing="xy")
    grid_points = center + grid_u[..., None] * u_axis + grid_v[..., None] * v_axis
    grid_dirs = np.broadcast_to(ray_axis.reshape(1, 1, 3), grid_points.shape).astype(np.float32)
    axes = np.stack([u_axis, v_axis, ray_axis], axis=0).astype(np.float32)

    for sensor_index, sensor in enumerate(getattr(env, "_tacmap_sensors", []) or []):
        slot = tacmap_slots[sensor_index] if sensor_index < len(tacmap_slots) else sensor_index
        if slot < 0 or slot >= sensor_count:
            continue
        actual_grid = _tacmap_sensor_local_grid(sensor, tac_rows, tac_cols)
        if actual_grid is None:
            tac_points[slot] = grid_points
            tac_dirs[slot] = grid_dirs
            tac_axes[slot] = axes
        else:
            tac_points[slot], tac_dirs[slot], tac_axes[slot] = actual_grid
        tac_valid[slot] = 1

    layout_arrays.update(
        {
            "tacmap_grid_points_l_m": tac_points,
            "tacmap_ray_directions_l": tac_dirs,
            "tacmap_grid_axes_l": tac_axes,
            "tacmap_grid_layout_valid": tac_valid,
        }
    )
    return layout_arrays


def cfg_pressure_grid_distances(cfg: IntegratedTactileEnvCfg) -> tuple[float, float]:
    point_distance = float(getattr(cfg, "point_distance", 0.0))
    row_distance = getattr(cfg, "row_distance", None)
    col_distance = getattr(cfg, "col_distance", None)
    row = point_distance if row_distance is None else float(row_distance)
    col = point_distance if col_distance is None else float(col_distance)
    return row, col


def build_normal_ray_runtime_alignment(
    layout_arrays: dict[str, np.ndarray],
    *,
    distance_mode: str,
    max_distance_m: float,
) -> dict[str, object]:
    required = (
        "pressure_taxel_points_l_m",
        "tacmap_grid_points_l_m",
        "tacmap_grid_axes_l",
    )
    missing = [key for key in required if key not in layout_arrays]
    if missing:
        raise RuntimeError(f"normal-ray pressure requires layout arrays: missing {missing}")

    source_index, valid_mask, nn_distance, summary = build_reference_to_pressure_alignment(
        layout_arrays["pressure_taxel_points_l_m"],
        layout_arrays["tacmap_grid_points_l_m"],
        reference_axes=layout_arrays.get("tacmap_grid_axes_l"),
        pressure_valid=layout_arrays.get("pressure_taxel_layout_valid"),
        reference_valid=layout_arrays.get("tacmap_grid_layout_valid"),
        distance_mode=str(distance_mode),
        max_distance_m=float(max_distance_m),
    )
    return {
        "source_index": source_index,
        "valid_mask": valid_mask,
        "nn_distance_m": nn_distance,
        "summary": summary,
    }


def _calibration_value_for_pressure_shape(value, shape: tuple[int, int, int]):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    sensors, rows, cols = shape
    if arr.size == rows * cols:
        return arr.astype(np.float32).reshape(1, rows, cols)
    if arr.size == sensors * rows * cols:
        return arr.astype(np.float32).reshape(sensors, rows, cols)
    return arr.astype(np.float32)


def normal_ray_pressure_calibration(cfg: IntegratedTactileEnvCfg, shape: tuple[int, int, int]) -> PressureCalibration:
    return PressureCalibration(
        gain=_calibration_value_for_pressure_shape(cfg.pressure_gain, shape),
        bias=_calibration_value_for_pressure_shape(cfg.pressure_bias, shape),
        stiffness=_calibration_value_for_pressure_shape(cfg.stiffness, shape),
        damping=_calibration_value_for_pressure_shape(cfg.damping, shape),
        max_force=_calibration_value_for_pressure_shape(cfg.max_force, shape),
        gamma=_calibration_value_for_pressure_shape(cfg.pressure_gamma, shape),
        threshold=_calibration_value_for_pressure_shape(cfg.pressure_threshold, shape),
        area=_calibration_value_for_pressure_shape(cfg.taxel_area, shape),
    )


def penetration_velocity_from_previous(
    penetration: np.ndarray,
    previous_penetration: np.ndarray | None,
    *,
    physics_dt: float,
) -> np.ndarray:
    current = np.asarray(penetration, dtype=np.float32)
    if previous_penetration is None or previous_penetration.shape != current.shape:
        return np.zeros_like(current, dtype=np.float32)
    dt = max(1.0e-9, float(physics_dt))
    return ((current - previous_penetration) / dt).astype(np.float32)


def normal_ray_pressure_output_from_penetration(
    penetration: np.ndarray,
    signed_distance: np.ndarray,
    velocity: np.ndarray,
    *,
    cfg: IntegratedTactileEnvCfg,
) -> dict[str, np.ndarray]:
    penetration = np.asarray(penetration, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    calibration = normal_ray_pressure_calibration(cfg, tuple(int(v) for v in penetration.shape))
    raw, pressure = calibrate_penetration(
        penetration,
        calibration,
        penetration_velocity=velocity,
        normalize=True,
    )
    raw = np.asarray(raw, dtype=np.float32)
    pressure = np.asarray(pressure, dtype=np.float32)
    total_force, center = pressure_map_stats(raw)
    return {
        "penetration_m": penetration,
        "signed_distance_m": np.asarray(signed_distance, dtype=np.float32),
        "penetration_velocity_mps": velocity,
        "pressure_raw_n": raw,
        "pressure_norm": pressure,
        "total_force_n": np.asarray(total_force, dtype=np.float32),
        "center_of_pressure_px": np.asarray(center, dtype=np.float32),
        "contact_mask": (penetration > 0.0).astype(np.uint8),
    }


def normal_ray_pressure_from_tacmap(
    tacmap_raw: np.ndarray | None,
    alignment: dict[str, object] | None,
    *,
    cfg: IntegratedTactileEnvCfg,
    previous_penetration: np.ndarray | None,
    contact_deadband_m: float,
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None]:
    if tacmap_raw is None or alignment is None:
        return None, previous_penetration

    aligned = apply_reference_alignment_to_values(
        np.asarray(tacmap_raw, dtype=np.float32)[None, ...],
        alignment["source_index"],
        alignment["valid_mask"],
    )[0]
    penetration = np.maximum(aligned - max(0.0, float(contact_deadband_m)), 0.0).astype(np.float32)
    velocity = penetration_velocity_from_previous(
        penetration,
        previous_penetration,
        physics_dt=float(getattr(cfg, "physics_dt", 0.0) or 0.0),
    )
    return (
        normal_ray_pressure_output_from_penetration(
            penetration,
            (-penetration).astype(np.float32),
            velocity,
            cfg=cfg,
        ),
        penetration,
    )


def load_usd_mesh_as_object_local_triangles(usd_path: str | Path, *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Load a USD mesh into object-local triangle arrays.

    Imports USD lazily so non-Isaac utility environments can still import and
    compile this runner.
    """

    from pxr import Gf, Usd, UsdGeom  # type: ignore[import-not-found]

    path = Path(usd_path).expanduser()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open geometry-normal-ray USD: {path}")

    xform_cache = UsdGeom.XformCache()
    vertices_parts: list[np.ndarray] = []
    triangles_parts: list[np.ndarray] = []
    mesh_paths: list[str] = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr().Get()
        face_counts_attr = mesh.GetFaceVertexCountsAttr().Get()
        face_indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
        if points_attr is None or face_counts_attr is None or face_indices_attr is None:
            continue

        matrix = xform_cache.GetLocalToWorldTransform(prim)
        points = np.asarray(
            [
                tuple(matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))))
                for p in points_attr
            ],
            dtype=np.float32,
        )
        face_counts = np.asarray(face_counts_attr, dtype=np.int32)
        face_indices = np.asarray(face_indices_attr, dtype=np.int32)
        triangles = _triangulate_face_indices(face_counts, face_indices)
        if points.size == 0 or triangles.size == 0:
            continue
        vertices_parts.append(points)
        triangles_parts.append(triangles + vertex_offset)
        mesh_paths.append(str(prim.GetPath()))
        vertex_offset += int(points.shape[0])

    if not vertices_parts or not triangles_parts:
        raise RuntimeError(f"No UsdGeom.Mesh triangles found in geometry-normal-ray USD: {path}")

    vertices = np.concatenate(vertices_parts, axis=0).astype(np.float32)
    triangles = np.concatenate(triangles_parts, axis=0).astype(np.int64)
    vertices *= float(scale)
    original_vertex_count = int(vertices.shape[0])
    vertices, triangles, weld_summary = weld_duplicate_triangle_vertices(vertices, triangles)
    summary = {
        "usd_path": str(path),
        "mesh_paths": mesh_paths,
        "vertex_count": int(vertices.shape[0]),
        "original_vertex_count": original_vertex_count,
        "triangle_count": int(triangles.shape[0]),
        "scale": float(scale),
        "bounds_min": [float(v) for v in np.min(vertices, axis=0)],
        "bounds_max": [float(v) for v in np.max(vertices, axis=0)],
        "weld": weld_summary,
    }
    summary["topology"] = triangle_mesh_topology_diagnostics(vertices, triangles)
    return vertices, triangles, summary


def load_npy_mesh_as_object_local_triangles(
    vertices_npy: str | Path,
    triangles_npy: str | Path,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    vertices_path = Path(vertices_npy).expanduser()
    triangles_path = Path(triangles_npy).expanduser()
    vertices = np.load(vertices_path).astype(np.float32)
    triangles = np.load(triangles_path).astype(np.int64)
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        raise ValueError(f"geometry vertices must have shape (V,3), got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[-1] != 3:
        raise ValueError(f"geometry triangles must have shape (T,3), got {triangles.shape}")
    vertices = (vertices * float(scale)).astype(np.float32)
    summary = {
        "source": "npy",
        "vertices_npy": str(vertices_path),
        "triangles_npy": str(triangles_path),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "scale": float(scale),
        "bounds_min": [float(v) for v in np.min(vertices, axis=0)],
        "bounds_max": [float(v) for v in np.max(vertices, axis=0)],
    }
    summary["topology"] = triangle_mesh_topology_diagnostics(vertices, triangles)
    return vertices, triangles, summary


def _triangulate_face_indices(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    if np.all(face_counts == 3) and face_indices.size % 3 == 0:
        return face_indices.reshape(-1, 3).astype(np.int64)
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for count in face_counts:
        n = int(count)
        if n >= 3:
            v0 = int(face_indices[cursor])
            for i in range(1, n - 1):
                triangles.append((v0, int(face_indices[cursor + i]), int(face_indices[cursor + i + 1])))
        cursor += n
    return np.asarray(triangles, dtype=np.int64)


def geometry_normal_ray_pressure_from_mesh(
    layout_arrays: dict[str, np.ndarray],
    *,
    object_vertices_l: np.ndarray | None,
    triangles: np.ndarray | None,
    object_pose_w: np.ndarray | None,
    touch_pose_w: np.ndarray | None,
    cfg: IntegratedTactileEnvCfg,
    previous_penetration: np.ndarray | None,
    rest_distance_m: float | np.ndarray | None,
    rest_baseline_m: np.ndarray | None,
    contact_deadband_m: float,
    max_distance_m: float,
    mode: str,
    taxel_sample_pattern: str = "center",
    taxel_sample_spacing_m: float | None = None,
    sample_aggregation: str = "max",
    sample_min_support_fraction: float = 0.0,
) -> tuple[dict[str, np.ndarray] | None, np.ndarray | None, np.ndarray | None]:
    if object_vertices_l is None or triangles is None:
        return None, previous_penetration, rest_baseline_m
    if not _pose_valid(object_pose_w) or not _pose_valid(touch_pose_w):
        return None, previous_penetration, rest_baseline_m
    if "pressure_taxel_points_l_m" not in layout_arrays or "pressure_taxel_normals_l" not in layout_arrays:
        return None, previous_penetration, rest_baseline_m

    pressure_points = np.asarray(layout_arrays["pressure_taxel_points_l_m"], dtype=np.float32)
    pressure_normals = np.asarray(layout_arrays["pressure_taxel_normals_l"], dtype=np.float32)
    if pressure_points.shape != pressure_normals.shape or pressure_points.ndim != 4 or pressure_points.shape[-1] != 3:
        return None, previous_penetration, rest_baseline_m
    ray_origins = np.asarray(layout_arrays.get("geometry_normal_ray_origin_points_l_m", pressure_points), dtype=np.float32)
    if ray_origins.shape != pressure_points.shape:
        ray_origins = pressure_points
    taxel_valid = np.asarray(
        layout_arrays.get(
            "geometry_normal_ray_origin_valid_mask",
            np.ones(pressure_points.shape[:3], dtype=np.uint8),
        ),
        dtype=bool,
    )
    if taxel_valid.shape != pressure_points.shape[:3]:
        taxel_valid = np.ones(pressure_points.shape[:3], dtype=bool)

    object_vertices_touch_l = transform_points_object_to_touch_l(
        object_vertices_l,
        object_pose_w=np.asarray(object_pose_w, dtype=np.float32),
        touch_pose_w=np.asarray(touch_pose_w, dtype=np.float32),
    )
    pressure_valid = np.asarray(
        layout_arrays.get("pressure_taxel_layout_valid", np.ones((pressure_points.shape[0],), dtype=np.uint8)),
        dtype=bool,
    )
    geometry_source = GeometryNormalRayPenetrationSource(
        rest_distance_m=0.0,
        contact_deadband_m=max(0.0, float(contact_deadband_m)),
        max_distance_m=float(max_distance_m),
    )
    sample_stats: dict[str, np.ndarray] = {}
    if str(mode) == "inside_exit":
        row_distance_m, col_distance_m = cfg_pressure_grid_distances(cfg)
        sample_offsets = geometry_normal_ray_taxel_sample_offsets(
            pressure_points,
            pressure_normals,
            pattern=taxel_sample_pattern,
            spacing_m=taxel_sample_spacing_m
            if taxel_sample_spacing_m is not None
            else min(row_distance_m, col_distance_m) * 0.5,
        )
        frame = geometry_source.frame_from_closed_triangle_mesh_inside(
            ray_origins,
            pressure_normals,
            vertices_l=object_vertices_touch_l,
            triangles=triangles,
            surface_sample_offsets_l=sample_offsets,
            sample_aggregation=sample_aggregation,
            sample_min_support_fraction=float(sample_min_support_fraction),
        )
        penetration = np.asarray(frame.penetration_m, dtype=np.float32).copy()
        for stat_key in (
            "sample_support_fraction",
            "sample_active_count",
            "sample_mean_penetration_m",
            "sample_positive_mean_penetration_m",
            "sample_max_penetration_m",
            "sample_penetrations_m",
            "sample_offsets_l",
            "sample_points_l_m",
        ):
            stat_value = getattr(frame, stat_key, None)
            if stat_value is not None:
                sample_stats[stat_key] = np.asarray(stat_value, dtype=np.float32).copy()
        for sensor_idx in range(min(penetration.shape[0], pressure_valid.shape[0])):
            if not bool(pressure_valid[sensor_idx]):
                penetration[sensor_idx] = 0.0
                for stat_value in sample_stats.values():
                    if stat_value.shape[: penetration.ndim] == penetration.shape:
                        stat_value[sensor_idx] = 0.0
        penetration = np.where(taxel_valid, penetration, 0.0).astype(np.float32)
        for key, stat_value in list(sample_stats.items()):
            if stat_value.shape[: taxel_valid.ndim] == taxel_valid.shape:
                mask = taxel_valid.reshape((*taxel_valid.shape, *([1] * (stat_value.ndim - taxel_valid.ndim))))
                sample_stats[key] = np.where(mask, stat_value, 0.0).astype(np.float32)
        frame_signed = (-penetration).astype(np.float32)
        next_rest_baseline = rest_baseline_m
    else:
        ray_distance, valid = geometry_source.ray_distance_to_triangle_mesh(
            ray_origins,
            pressure_normals,
            vertices_l=object_vertices_touch_l,
            triangles=triangles,
        )
        valid = valid & taxel_valid
        ray_distance = np.where(taxel_valid, ray_distance, np.inf).astype(np.float32)
        for sensor_idx in range(min(ray_distance.shape[0], pressure_valid.shape[0])):
            if not bool(pressure_valid[sensor_idx]):
                valid[sensor_idx] = False
                ray_distance[sensor_idx] = np.inf

        if rest_distance_m is None:
            if rest_baseline_m is None or rest_baseline_m.shape != ray_distance.shape:
                rest = np.where(np.isfinite(ray_distance), ray_distance, float(max_distance_m)).astype(np.float32)
            else:
                rest = rest_baseline_m
        else:
            rest = rest_distance_m

        frame = NormalRayPenetrationSource(
            rest_distance_m=rest,
            contact_deadband_m=max(0.0, float(contact_deadband_m)),
        ).frame_from_ray_distance(ray_distance, valid_mask=valid)
        penetration = np.asarray(frame.penetration_m, dtype=np.float32)
        frame_signed = np.asarray(frame.signed_distance_m, dtype=np.float32)
        next_rest_baseline = np.asarray(rest, dtype=np.float32) if rest_distance_m is None else rest_baseline_m

    velocity = penetration_velocity_from_previous(
        penetration,
        previous_penetration,
        physics_dt=float(getattr(cfg, "physics_dt", 0.0) or 0.0),
    )
    output = normal_ray_pressure_output_from_penetration(
        penetration,
        frame_signed,
        velocity,
        cfg=cfg,
    )
    output.update(sample_stats)
    return output, penetration, next_rest_baseline


def geometry_normal_ray_taxel_sample_offsets(
    pressure_points: np.ndarray,
    pressure_normals: np.ndarray,
    *,
    pattern: str,
    spacing_m: float,
) -> np.ndarray | None:
    pattern = str(pattern).lower()
    if pattern == "center":
        return None
    spacing = max(0.0, float(spacing_m))
    if spacing <= 0.0:
        return None

    points = np.asarray(pressure_points, dtype=np.float32)
    normals = np.asarray(pressure_normals, dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        return None
    if normals.shape != points.shape:
        return None

    row_axis, col_axis = pressure_taxel_grid_axes(points, normals)
    if pattern == "cross_5":
        coeffs = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    elif pattern == "grid_3x3":
        coeffs = np.asarray([(r, c) for r in (-1.0, 0.0, 1.0) for c in (-1.0, 0.0, 1.0)], dtype=np.float32)
    else:
        raise ValueError("geometry normal-ray taxel sample pattern must be 'center', 'cross_5', or 'grid_3x3'")

    offsets = (
        coeffs[None, None, None, :, 0, None] * row_axis[:, None, None, None, :]
        + coeffs[None, None, None, :, 1, None] * col_axis[:, None, None, None, :]
    )
    return (spacing * offsets).astype(np.float32)


def pressure_taxel_grid_axes(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sensor_count = int(points.shape[0])
    row_axes = np.zeros((sensor_count, 3), dtype=np.float32)
    col_axes = np.zeros((sensor_count, 3), dtype=np.float32)
    for sensor_idx in range(sensor_count):
        normal = _mean_unit_vector(normals[sensor_idx].reshape(-1, 3), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        row = _mean_unit_vector(
            (points[sensor_idx, 1:, :, :] - points[sensor_idx, :-1, :, :]).reshape(-1, 3)
            if points.shape[1] > 1
            else np.zeros((0, 3), dtype=np.float32),
            fallback=None,
        )
        col = _mean_unit_vector(
            (points[sensor_idx, :, 1:, :] - points[sensor_idx, :, :-1, :]).reshape(-1, 3)
            if points.shape[2] > 1
            else np.zeros((0, 3), dtype=np.float32),
            fallback=None,
        )
        if row is None:
            row = _orthogonal_unit_vector(normal)
        row = _normalize_vector(row - normal * float(np.dot(row, normal)), fallback=_orthogonal_unit_vector(normal))
        if col is None:
            col = np.cross(normal, row)
        col = _normalize_vector(col - normal * float(np.dot(col, normal)), fallback=np.cross(normal, row))
        row_axes[sensor_idx] = row
        col_axes[sensor_idx] = col
    return row_axes, col_axes


def _mean_unit_vector(values: np.ndarray, *, fallback: np.ndarray | None) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return fallback
    finite = arr[np.all(np.isfinite(arr), axis=1)]
    finite = finite[np.linalg.norm(finite, axis=1) > 1.0e-12]
    if finite.size == 0:
        return fallback
    return _normalize_vector(np.mean(finite, axis=0), fallback=fallback)


def _normalize_vector(value: np.ndarray, *, fallback: np.ndarray | None) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1.0e-12:
        return (vec / norm).astype(np.float32)
    if fallback is not None:
        fb = np.asarray(fallback, dtype=np.float32)
        fb_norm = max(float(np.linalg.norm(fb)), 1.0e-12)
        return (fb / fb_norm).astype(np.float32)
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _optional_vec3(value, *, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be a finite 3-vector")
    return tuple(float(v) for v in arr)


def _optional_unit_quat_wxyz(value, *, field_name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-12:
        raise ValueError(f"{field_name} must have non-zero length")
    arr = arr / norm
    return tuple(float(v) for v in arr)


def _safe_output_name(path: Path) -> str:
    raw = path.name
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    return safe.strip("_") or "robot"


def default_robot_usd_output_dir(robot_urdf_path: Path) -> Path:
    digest = hashlib.sha256(robot_urdf_path.read_bytes()).hexdigest()[:12]
    return OFFICIAL_REPLAY_ROOT / "output" / f"{_safe_output_name(robot_urdf_path)}_{digest}_usd"


def apply_press_motion_cli_overrides(cfg: IntegratedTactileEnvCfg) -> None:
    center_l = _optional_vec3(args.press_center_l, field_name="--press-center-l")
    center_offset_l = _optional_vec3(args.press_center_offset_l, field_name="--press-center-offset-l")
    normal_l = _optional_vec3(args.press_normal_l, field_name="--press-normal-l")

    if center_l is not None:
        cfg.press_center_l = center_l
    if center_offset_l is not None:
        base = np.asarray(cfg.press_center_l if cfg.press_center_l is not None else (0.0, 0.0, 0.0), dtype=np.float32)
        cfg.press_center_l = tuple(float(v) for v in base + np.asarray(center_offset_l, dtype=np.float32))
    if normal_l is not None:
        cfg.press_normal_l = tuple(
            float(v)
            for v in _normalize_vector(np.asarray(normal_l, dtype=np.float32), fallback=np.array([0.0, 0.0, 1.0]))
        )


class PressLiveControl:
    """Hot-reload press-motion controls from a small JSON file."""

    def __init__(self, path: str | Path, cfg: IntegratedTactileEnvCfg, *, write_template: bool = False):
        self.path = Path(path).expanduser()
        self._last_mtime_ns: int | None = None
        self._last_error: str | None = None
        if write_template or not self.path.exists():
            self.write_template(cfg)

    def write_template(self, cfg: IntegratedTactileEnvCfg) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        template = {
            "enabled": True,
            "center_l": None,
            "center_offset_l": [0.0, 0.0, 0.0],
            "world_pos": None,
            "world_quat_wxyz": None,
            "normal_l": list(cfg.press_normal_l) if cfg.press_normal_l is not None else None,
            "offset_m": None,
            "slide_offset_l": [0.0, 0.0, 0.0],
            "notes": (
                "Edit while the GUI is running. world_pos/world_quat_wxyz move /World/Plug in world frame. "
                "Other vector fields use meters in press_touch_link local frame. "
                "offset_m overrides the scripted press depth when set; negative values press into the pad."
            ),
        }
        self.path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def apply(self, env) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._last_error != "missing":
                print(f"[WARN] live control file missing: {self.path}", flush=True)
                self._last_error = "missing"
            return
        if self._last_mtime_ns == stat.st_mtime_ns:
            return
        self._last_mtime_ns = stat.st_mtime_ns
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not bool(data.get("enabled", True)):
                env.set_press_motion_control()
                print("[INFO] live press control disabled; using scripted press motion", flush=True)
                self._last_error = None
                return
            center_l = _optional_vec3(data.get("center_l"), field_name="live_control.center_l")
            center_offset_l = _optional_vec3(
                data.get("center_offset_l"), field_name="live_control.center_offset_l"
            )
            normal_l = _optional_vec3(data.get("normal_l"), field_name="live_control.normal_l")
            slide_offset_l = _optional_vec3(
                data.get("slide_offset_l"), field_name="live_control.slide_offset_l"
            )
            world_pos = _optional_vec3(data.get("world_pos"), field_name="live_control.world_pos")
            world_quat = _optional_unit_quat_wxyz(
                data.get("world_quat_wxyz"), field_name="live_control.world_quat_wxyz"
            )
            if world_pos is not None:
                setter = getattr(env, "set_press_object_world_pose", None)
                if callable(setter):
                    setter(world_pos, world_quat)
            offset_raw = data.get("offset_m", data.get("press_offset_m"))
            offset_m = None if offset_raw is None else float(offset_raw)
            env.set_press_motion_control(
                center_l=center_l,
                center_offset_l=center_offset_l,
                normal_l=normal_l,
                offset_m=offset_m,
                slide_offset_l=slide_offset_l,
            )
            print(
                "[INFO] live press control applied: "
                f"center_l={center_l} center_offset_l={center_offset_l} normal_l={normal_l} "
                f"offset_m={offset_m} slide_offset_l={slide_offset_l} world_pos={world_pos}",
                flush=True,
            )
            self._last_error = None
        except Exception as exc:
            message = str(exc)
            if self._last_error != message:
                print(f"[WARN] failed to load live control file {self.path}: {message}", flush=True)
                self._last_error = message


class PresserPoseRecorder:
    """Persist the current /World/Plug world pose for GUI placement tuning."""

    def __init__(self, path: str | Path, cfg: IntegratedTactileEnvCfg, *, presser: str, save_every: int = 5):
        self.path = Path(path).expanduser()
        self.save_every = max(1, int(save_every))
        self._cfg = cfg
        self._presser = str(presser)
        self._last_signature: tuple[float, ...] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def maybe_write(self, env, obs: dict | None, step: int, *, force: bool = False) -> None:
        if not force and step >= 0 and step % self.save_every != 0:
            return
        pose = self._read_pose(env, obs)
        if not _pose_valid(pose):
            return
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        quat_norm = float(np.linalg.norm(pose[3:7]))
        if quat_norm <= 1.0e-12:
            return
        pose[3:7] = pose[3:7] / quat_norm
        signature = tuple(float(f"{value:.9g}") for value in pose)
        if not force and signature == self._last_signature:
            return
        self._last_signature = signature

        x, y, z = (float(value) for value in pose[:3])
        qw, qx, qy, qz = (float(value) for value in pose[3:7])
        cli = (
            "--press-object-control manual_gui "
            f"--presser-world-pos {x:.9g} {y:.9g} {z:.9g} "
            f"--presser-world-quat-wxyz {qw:.9g} {qx:.9g} {qy:.9g} {qz:.9g}"
        )
        payload = {
            "schema_version": "presser_pose_v1",
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
            "step": int(step),
            "prim_path": "/World/Plug",
            "press_object_control": str(getattr(self._cfg, "press_object_control", "")),
            "presser": self._presser,
            "world_position_xyz_m": [x, y, z],
            "world_quaternion_wxyz": [qw, qx, qy, qz],
            "world_pose_xyz_xyzw": [x, y, z, qx, qy, qz, qw],
            "reuse_cli": cli,
            "notes": (
                "This is the current /World/Plug root pose. Reuse it with "
                "--press-object-control manual_gui; initial_only will recompute and overwrite the scripted initial pose."
            ),
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read_pose(env, obs: dict | None) -> np.ndarray | None:
        getter = getattr(env, "press_object_pose_w_numpy", None)
        if callable(getter):
            pose = getter()
            if _pose_valid(pose):
                return np.asarray(pose, dtype=np.float32).reshape(7)
        if obs is not None:
            pose = obs.get("plug_pose")
            if _pose_valid(pose):
                return np.asarray(pose, dtype=np.float32).reshape(7)
        return None


class PressDebugLogger:
    """Write compact JSONL telemetry for human-in-the-loop GUI press debugging."""

    def __init__(self, path: str | Path, cfg: IntegratedTactileEnvCfg, *, every: int = 5):
        self.path = Path(path).expanduser()
        self.every = max(1, int(every))
        self._cfg = cfg
        self._baseline: dict[str, object] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self.write_event("created", extra={"path": str(self.path)})

    def close(self) -> None:
        if not self._file.closed:
            self.write_event("closed")
            self._file.close()

    def reset_baseline(self) -> None:
        self._baseline.clear()

    def write_event(self, event: str, env=None, obs: dict | None = None, extra: dict[str, object] | None = None) -> None:
        payload = self._snapshot(env, obs, phase="event", step=None)
        payload["event"] = str(event)
        if extra:
            payload["extra"] = extra
        self._write(payload)

    def maybe_write(
        self,
        *,
        phase: str,
        step: int,
        env,
        obs: dict | None = None,
        force: bool = False,
    ) -> None:
        if not force and int(step) >= 0 and int(step) % self.every != 0:
            return
        self._write(self._snapshot(env, obs, phase=phase, step=int(step)))

    def _write(self, payload: dict[str, object]) -> None:
        self._file.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def _snapshot(self, env, obs: dict | None, *, phase: str, step: int | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "press_debug_v1",
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "phase": str(phase),
            "step": step,
            "cfg": {
                "mode": str(args.mode),
                "legacy_finger_arg": str(args.finger),
                "fingers_arg": str(args.fingers or ""),
                "focus_finger_arg": str(args.focus_finger or ""),
                "press_touch_link": str(getattr(self._cfg, "press_touch_link", "")),
                "press_touch_links": [str(value) for value in getattr(self._cfg, "press_touch_links", ())],
                "press_object_control": str(getattr(self._cfg, "press_object_control", "")),
                "press_motion_actor": str(getattr(self._cfg, "press_motion_actor", "")),
                "press_hand_axis_link": str(getattr(self._cfg, "press_hand_axis_link", "")),
                "press_hand_axis_l": [
                    float(value) for value in getattr(self._cfg, "press_hand_axis_l", (1.0, 0.0, 0.0))
                ],
                "press_finger_joints": [
                    str(value) for value in getattr(self._cfg, "press_finger_joints", ())
                ],
                "press_finger_start_rad": float(getattr(self._cfg, "press_finger_start_rad", 0.0)),
                "press_finger_end_rad": float(getattr(self._cfg, "press_finger_end_rad", 0.0)),
                "press_finger_kinematic_sensor_pose": bool(
                    getattr(self._cfg, "press_finger_kinematic_sensor_pose", True)
                ),
                "press_start_offset": float(getattr(self._cfg, "press_start_offset", 0.0)),
                "press_end_offset": float(getattr(self._cfg, "press_end_offset", 0.0)),
                "presser_collision_enabled": bool(
                    getattr(self._cfg, "press_object_collision_enabled", True)
                ),
                "press_distance": float(
                    getattr(self._cfg, "press_start_offset", 0.0)
                    - getattr(self._cfg, "press_end_offset", 0.0)
                ),
                "press_indent_depth": (
                    None
                    if getattr(self._cfg, "press_indent_depth", None) is None
                    else float(getattr(self._cfg, "press_indent_depth"))
                ),
                "press_contact_search_distance": (
                    None
                    if getattr(self._cfg, "press_contact_search_distance", None) is None
                    else float(getattr(self._cfg, "press_contact_search_distance"))
                ),
                "press_contact_threshold": float(getattr(self._cfg, "press_contact_threshold", 1.0e-7)),
                "press_steps": int(getattr(self._cfg, "press_steps", 0)),
                "press_hold_joint_pose": str(getattr(self._cfg, "press_hold_joint_pose", "")),
            },
        }
        if env is None:
            return payload

        payload["runtime"] = {
            "press_counter": int(getattr(env, "_press_counter", -1)),
            "press_touch_body_idx": _safe_int(getattr(env, "_press_touch_body_idx", None)),
            "press_motion_actor": _call_or_none(getattr(env, "_press_motion_actor", None)),
            "press_object_control_mode": _call_or_none(getattr(env, "_press_object_control_mode", None)),
            "press_finger_joint_ids": [
                int(value) for value in getattr(env, "_press_finger_joint_ids", [])
            ],
            "press_finger_joint_names": [
                str(value) for value in getattr(env, "_press_finger_joint_names", [])
            ],
            "selected_links": [str(value) for value in getattr(env, "_selected_links", [])],
            "sensor_slot_order": [int(value) for value in getattr(env, "_sensor_slot_order", [])],
        }
        plug_pose = self._pose_from_env(env, obs, key="plug_pose", getter_name="press_object_pose_w_numpy")
        touch_pose = self._pose_from_env(env, obs, key="press_touch_pose", getter_name="_press_touch_pose_numpy")
        payload["poses"] = {
            "plug_w": _pose_payload(plug_pose),
            "press_touch_w": _pose_payload(touch_pose),
        }

        robot = getattr(env, "_robot", None)
        if robot is not None:
            joint_payload, joint_summary = self._joint_payload(robot)
            body_payload, body_summary = self._body_payload(robot)
            payload["joints"] = joint_payload
            payload["bodies"] = body_payload
            payload["motion_summary"] = {
                **joint_summary,
                **body_summary,
                "plug_translation_from_first_m": _translation_delta(
                    self._baseline_value("plug_pose", plug_pose), plug_pose
                ),
                "press_touch_translation_from_first_m": _translation_delta(
                    self._baseline_value("press_touch_pose", touch_pose), touch_pose
                ),
            }

        force_arr = None
        penetration_arr = None
        signed_distance_arr = None
        force = None if obs is None else obs.get("pressure_force_map", obs.get("tactile"))
        if force is not None:
            force_arr = np.asarray(force, dtype=np.float32)
            payload["pressure"] = {
                "shape": [int(value) for value in force_arr.shape],
                "labels": pressure_sensor_labels_from_env(
                    env,
                    int(force_arr.shape[0]),
                    fallback_label=str(getattr(self._cfg, "press_touch_link", args.finger)),
                ),
                "max": float(np.nanmax(force_arr)) if force_arr.size else 0.0,
                "active_count": int(np.count_nonzero(force_arr > float(args.active_threshold))),
                "sum": float(np.nansum(force_arr)) if force_arr.size else 0.0,
            }
        penetration = None if obs is None else obs.get("pressure_penetration_map")
        if penetration is not None:
            penetration_arr = np.asarray(penetration, dtype=np.float32)
            payload["pressure_penetration"] = {
                "shape": [int(value) for value in penetration_arr.shape],
                "max_m": float(np.nanmax(penetration_arr)) if penetration_arr.size else 0.0,
                "active_count": int(np.count_nonzero(penetration_arr > 0.0)),
                "sum_m": float(np.nansum(penetration_arr)) if penetration_arr.size else 0.0,
            }
        payload["press_depth"] = press_depth_summary_from_env(env, penetration_arr, plug_pose=plug_pose)
        signed_distance = None if obs is None else obs.get("pressure_signed_distance_map")
        if signed_distance is not None:
            signed_distance_arr = np.asarray(signed_distance, dtype=np.float32)
            payload["pressure_signed_distance"] = {
                "shape": [int(value) for value in signed_distance_arr.shape],
                "min_m": float(np.nanmin(signed_distance_arr)) if signed_distance_arr.size else 0.0,
                "max_m": float(np.nanmax(signed_distance_arr)) if signed_distance_arr.size else 0.0,
            }
        if force_arr is not None:
            payload["pressure_active_taxels"] = self._active_taxel_payload(
                env,
                force_arr,
                penetration_arr=penetration_arr,
                signed_distance_arr=signed_distance_arr,
            )
        return payload

    def _baseline_value(self, key: str, value):
        if value is None:
            return None
        if key not in self._baseline:
            self._baseline[key] = np.asarray(value, dtype=np.float64).copy()
        return self._baseline[key]

    @staticmethod
    def _pose_from_env(env, obs: dict | None, *, key: str, getter_name: str) -> np.ndarray | None:
        if obs is not None and key in obs and _pose_valid(obs[key]):
            return np.asarray(obs[key], dtype=np.float64).reshape(7)
        getter = getattr(env, getter_name, None)
        if callable(getter):
            try:
                pose = getter()
            except Exception:
                pose = None
            if _pose_valid(pose):
                return np.asarray(pose, dtype=np.float64).reshape(7)
        return None

    def _joint_payload(self, robot) -> tuple[dict[str, object], dict[str, float]]:
        names = [str(name) for name in getattr(robot, "joint_names", [])]
        data = getattr(robot, "data", None)
        pos = _tensor_row_to_numpy(getattr(data, "joint_pos", None))
        vel = _tensor_row_to_numpy(getattr(data, "joint_vel", None))
        target = _tensor_row_to_numpy(getattr(data, "joint_pos_target", None))
        out: dict[str, object] = {}
        max_delta = {"middle": 0.0, "index": 0.0}
        for i, name in enumerate(names):
            group = _finger_group(name)
            if group not in {"middle", "index"}:
                continue
            value = float(pos[i]) if pos is not None and i < pos.size else None
            baseline_key = f"joint:{name}"
            baseline = self._baseline_value(baseline_key, [value] if value is not None else None)
            delta = None
            if baseline is not None and value is not None:
                delta = abs(float(value) - float(np.asarray(baseline).reshape(-1)[0]))
                max_delta[group] = max(max_delta[group], float(delta))
            out[name] = {
                "joint_group": group,
                "pos_rad": value,
                "vel_rad_s": float(vel[i]) if vel is not None and i < vel.size else None,
                "target_rad": float(target[i]) if target is not None and i < target.size else None,
                "abs_delta_from_first_rad": delta,
            }
        return out, {
            "max_tracked_middle_joint_delta_rad": max_delta["middle"],
            "max_tracked_index_joint_delta_rad": max_delta["index"],
        }

    def _body_payload(self, robot) -> tuple[dict[str, object], dict[str, float]]:
        names = [str(name) for name in getattr(robot, "body_names", [])]
        state = _tensor_row_to_numpy(getattr(getattr(robot, "data", None), "body_link_state_w", None), ndim=3)
        out: dict[str, object] = {}
        max_delta = {"middle": 0.0, "index": 0.0}
        if state is None:
            return out, {"max_tracked_middle_body_delta_m": 0.0, "max_tracked_index_body_delta_m": 0.0}
        for i, name in enumerate(names):
            group = _finger_group(name)
            if group not in {"middle", "index"} or i >= state.shape[0]:
                continue
            pose = np.asarray(state[i, :7], dtype=np.float64)
            baseline = self._baseline_value(f"body:{name}", pose)
            delta = _translation_delta(baseline, pose)
            if delta is not None:
                max_delta[group] = max(max_delta[group], float(delta))
            out[name] = {
                "body_group": group,
                "pose_w": _pose_payload(pose),
                "translation_from_first_m": delta,
            }
        return out, {
            "max_tracked_middle_body_delta_m": max_delta["middle"],
            "max_tracked_index_body_delta_m": max_delta["index"],
        }

    def _active_taxel_payload(
        self,
        env,
        force_arr: np.ndarray,
        *,
        penetration_arr: np.ndarray | None,
        signed_distance_arr: np.ndarray | None,
    ) -> list[dict[str, object]]:
        force = _ensure_sensor_grid(force_arr)
        penetration = _ensure_sensor_grid(penetration_arr)
        signed_distance = _ensure_sensor_grid(signed_distance_arr)
        layouts = self._layout_by_slot(env)
        out: list[dict[str, object]] = []
        for sensor_id in range(force.shape[0]):
            values = np.nan_to_num(force[sensor_id], nan=0.0, posinf=0.0, neginf=0.0)
            pen = penetration[sensor_id] if penetration is not None and sensor_id < penetration.shape[0] else None
            sdf = (
                signed_distance[sensor_id]
                if signed_distance is not None and sensor_id < signed_distance.shape[0]
                else None
            )
            active = values > float(args.active_threshold)
            if not np.any(active) and pen is not None:
                active = pen > 0.0
            active_rows, active_cols = np.nonzero(active)
            entry: dict[str, object] = {
                "sensor_id": int(sensor_id),
                "active_count": int(active_rows.size),
                "force_max": float(values.max()) if values.size else 0.0,
                "top_taxels": [],
            }
            if active_rows.size > 0:
                positive = np.where(active, values, 0.0)
                total = float(positive.sum())
                if total > 0.0:
                    rows, cols = np.indices(values.shape)
                    entry["center_of_pressure_rc"] = [
                        float((rows * positive).sum() / total),
                        float((cols * positive).sum() / total),
                    ]
                else:
                    entry["center_of_pressure_rc"] = [float(np.mean(active_rows)), float(np.mean(active_cols))]
                entry["bbox_rc"] = [
                    int(active_rows.min()),
                    int(active_cols.min()),
                    int(active_rows.max()),
                    int(active_cols.max()),
                ]

                rank_values = values.copy()
                if pen is not None and float(rank_values.max()) <= 0.0:
                    rank_values = pen.copy()
                rank_values = np.where(active, rank_values, -np.inf)
                order = np.argsort(rank_values.reshape(-1))[::-1]
                layout = layouts.get(sensor_id, {})
                points_l = layout.get("points_l")
                normals_l = layout.get("normals_l")
                top: list[dict[str, object]] = []
                for flat_idx in order[:8]:
                    row, col = np.unravel_index(int(flat_idx), values.shape)
                    if not np.isfinite(rank_values[row, col]):
                        continue
                    item: dict[str, object] = {
                        "row": int(row),
                        "col": int(col),
                        "force": float(values[row, col]),
                    }
                    if pen is not None:
                        item["penetration_m"] = float(pen[row, col])
                    if sdf is not None:
                        item["signed_distance_m"] = float(sdf[row, col])
                    if isinstance(points_l, np.ndarray) and points_l.shape[:2] == values.shape:
                        item["point_l_m"] = [float(v) for v in points_l[row, col].reshape(3)]
                    if isinstance(normals_l, np.ndarray) and normals_l.shape[:2] == values.shape:
                        item["normal_l"] = [float(v) for v in normals_l[row, col].reshape(3)]
                    top.append(item)
                entry["top_taxels"] = top
            out.append(entry)
        return out

    def _layout_by_slot(self, env) -> dict[int, dict[str, object]]:
        layouts: dict[int, dict[str, object]] = {}
        sensors = list(getattr(env, "_tactile_sensors", []) or [])
        slots = list(getattr(env, "_sensor_slot_order", []) or [])
        for sensor_idx, sensor in enumerate(sensors):
            slot = int(slots[sensor_idx]) if sensor_idx < len(slots) else int(sensor_idx)
            maps = tuple(getattr(sensor, "pressure_taxel_maps", ()) or ())
            if not maps:
                continue
            taxel_map = maps[0]
            rows, cols = (int(v) for v in getattr(taxel_map, "image_shape", (0, 0)))
            if rows <= 0 or cols <= 0:
                continue
            points_l = _backend_array_to_numpy(getattr(taxel_map, "points_l", None))
            normals_l = _backend_array_to_numpy(getattr(taxel_map, "normals_l", None))
            layouts[slot] = {
                "link_name": str(getattr(taxel_map, "link_name", "")),
                "points_l": points_l.reshape(rows, cols, 3) if points_l is not None else None,
                "normals_l": normals_l.reshape(rows, cols, 3) if normals_l is not None else None,
            }
        return layouts


def _safe_int(value) -> int | None:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _ensure_sensor_grid(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim == 3:
        return arr
    return None


def _backend_array_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _call_or_none(func) -> object | None:
    if not callable(func):
        return None
    try:
        return func()
    except Exception:
        return None


def _finger_group(name: str) -> str | None:
    lower = str(name).lower()
    if "right_index" in lower or "_index" in lower:
        return "index"
    if "right_mid" in lower or "_mid" in lower or "right_middle" in lower or "_middle" in lower:
        return "middle"
    return None


def _tensor_row_to_numpy(value, *, ndim: int = 2) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = value.detach().cpu().numpy()
    except Exception:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if ndim == 3:
        return np.asarray(arr[0], dtype=np.float64) if arr.ndim >= 3 else None
    return np.asarray(arr[0], dtype=np.float64).reshape(-1) if arr.ndim >= 2 else np.asarray(arr, dtype=np.float64)


def _pose_payload(pose: np.ndarray | None) -> dict[str, object] | None:
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size < 7:
        return None
    return {
        "pos_xyz_m": [float(value) for value in arr[:3]],
        "quat_wxyz": [float(value) for value in arr[3:7]],
    }


def _translation_delta(baseline, pose) -> float | None:
    if baseline is None or pose is None:
        return None
    a = np.asarray(baseline, dtype=np.float64).reshape(-1)
    b = np.asarray(pose, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return None
    return float(np.linalg.norm(b[:3] - a[:3]))


def wait_for_press_setup_before_motion(
    env,
    simulation_app,
    pose_recorder: PresserPoseRecorder | None,
    debug_logger: PressDebugLogger | None = None,
) -> None:
    if not bool(args.press_setup_before_motion):
        return

    start_path = Path(args.press_setup_start_file).expanduser() if args.press_setup_start_file else None
    enter_event = threading.Event()
    eof_event = threading.Event()
    prepare = getattr(env, "prepare_press_object_manual_setup", None)
    prepared_pose = prepare() if callable(prepare) else None

    def _wait_for_enter() -> None:
        try:
            input()
            enter_event.set()
        except EOFError:
            eof_event.set()

    if sys.stdin is not None and sys.stdin.isatty():
        thread = threading.Thread(target=_wait_for_enter, daemon=True)
        thread.start()
        enter_text = "press Enter in this terminal"
    elif start_path is None:
        print("[WARN] --press-setup-before-motion requested without interactive stdin; starting immediately.", flush=True)
        enter_event.set()
        enter_text = "stdin unavailable"
    else:
        enter_text = "stdin unavailable"

    trigger_text = enter_text
    if start_path is not None:
        trigger_text = f"{enter_text} or create {start_path}"
    robot_text = " Select /World/Robot to drag/rotate the hand;" if bool(args.setup_edit_robot_pose) else ""
    print(
        "[SETUP] Edit the world now."
        f"{robot_text} select /World/Plug to drag/rotate the presser; "
        f"then {trigger_text} to start the press motion.",
        flush=True,
    )
    if prepared_pose is not None:
        pos = " ".join(f"{float(value):.6g}" for value in prepared_pose[:3])
        quat = " ".join(f"{float(value):.6g}" for value in prepared_pose[3:7])
        print(f"[SETUP] Scripted initial pose synced to stage: pos=[{pos}] quat_wxyz=[{quat}]", flush=True)
    if debug_logger is not None:
        debug_logger.write_event("setup_started", env, extra={"prepared_pose": _pose_payload(prepared_pose)})
    if str(getattr(args, "press_object_control", "scripted")) != "manual_gui":
        print(
            "[SETUP] Note: after setup starts, live GUI dragging of /World/Plug requires "
            "--press-object-control manual_gui. The current mode will keep driving /World/Plug from code.",
            flush=True,
        )

    setup_step = 0
    sync_robot_hold = getattr(env, "sync_press_robot_hold_pose", None)
    sync_robot_root = getattr(env, "sync_press_robot_root_pose", None)
    sync_press_object = getattr(env, "capture_press_object_pose_as_motion_start", None)
    print_coordinates = bool(args.print_press_coordinates)
    coordinate_log_every = max(1, int(args.press_coordinate_log_every))

    while simulation_app.is_running() and not enter_event.is_set():
        if start_path is not None and start_path.exists():
            break
        if eof_event.is_set() and start_path is None:
            break
        if callable(sync_press_object):
            sync_press_object()
        if callable(sync_robot_hold):
            sync_robot_hold()
        if callable(sync_robot_root):
            sync_robot_root()
        simulation_app.update()
        if print_coordinates and setup_step % coordinate_log_every == 0:
            print_press_coordinates(env, phase="setup", step=setup_step)
        if callable(sync_press_object):
            sync_press_object()
        if callable(sync_robot_hold):
            sync_robot_hold()
        if callable(sync_robot_root):
            sync_robot_root()
        if pose_recorder is not None:
            pose_recorder.maybe_write(env, None, setup_step)
        if debug_logger is not None:
            debug_logger.maybe_write(phase="setup", step=setup_step, env=env)
        setup_step += 1
        time.sleep(1.0 / 60.0)

    capture = getattr(env, "capture_press_object_pose_as_motion_start", None)
    pose = capture() if callable(capture) else None
    if print_coordinates:
        print_press_coordinates(env, phase="setup_capture", step=setup_step)
    if pose is None:
        print("[WARN] Could not capture /World/Plug pose after setup; press motion will use scripted pose.", flush=True)
        if debug_logger is not None:
            debug_logger.write_event("setup_capture_failed", env, extra={"setup_steps": int(setup_step)})
        return
    if pose_recorder is not None:
        pose_recorder.maybe_write(env, None, setup_step, force=True)
    pos = " ".join(f"{float(value):.6g}" for value in pose[:3])
    quat = " ".join(f"{float(value):.6g}" for value in pose[3:7])
    print(f"[SETUP] Captured press motion start pose: pos=[{pos}] quat_wxyz=[{quat}]", flush=True)
    if debug_logger is not None:
        debug_logger.write_event(
            "setup_finished",
            env,
            extra={"setup_steps": int(setup_step), "captured_pose": _pose_payload(pose)},
        )


def _orthogonal_unit_vector(normal: np.ndarray) -> np.ndarray:
    n = _normalize_vector(normal, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return _normalize_vector(np.cross(n, ref), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))


def transform_points_object_to_touch_l(
    points_object_l: np.ndarray,
    *,
    object_pose_w: np.ndarray,
    touch_pose_w: np.ndarray,
) -> np.ndarray:
    object_pos = np.asarray(object_pose_w[:3], dtype=np.float32)
    object_quat = np.asarray(object_pose_w[3:7], dtype=np.float32)
    touch_pos = np.asarray(touch_pose_w[:3], dtype=np.float32)
    touch_quat = np.asarray(touch_pose_w[3:7], dtype=np.float32)
    points_w = object_pos[None, :] + quat_apply_np(object_quat, np.asarray(points_object_l, dtype=np.float32))
    return quat_apply_inverse_np(touch_quat, points_w - touch_pos[None, :]).astype(np.float32)


def quat_apply_np(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    v = np.asarray(vectors, dtype=np.float32)
    qvec = q[1:4]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def quat_apply_inverse_np(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    return quat_apply_np(conj, vectors)


def _pose_valid(pose: np.ndarray | None) -> bool:
    if pose is None:
        return False
    arr = np.asarray(pose, dtype=np.float32)
    return arr.shape == (7,) and np.all(np.isfinite(arr)) and float(np.linalg.norm(arr[3:7])) > 0.5


def quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rpy_to_quat_wxyz(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    )


def _is_fingertip_pressure_link(link_name: str) -> bool:
    name = str(link_name).lower()
    return "fingertip" in name or "_tip" in name or "tip_" in name


def _apply_finger_link_surface_defaults(cfg: IntegratedTactileEnvCfg, spec: dict[str, object]) -> None:
    link_surface = spec.get("link_surface")
    if not isinstance(link_surface, dict):
        return
    if not cli_arg_provided("--link-surface-ray-axis"):
        cfg.tacmap_link_surface_ray_axis = str(link_surface["ray_axis"])
    if not cli_arg_provided("--link-surface-grid-u-axis"):
        cfg.tacmap_link_surface_grid_u_axis = str(link_surface["grid_u_axis"])
    if not cli_arg_provided("--link-surface-grid-v-axis"):
        cfg.tacmap_link_surface_grid_v_axis = str(link_surface["grid_v_axis"])
    if not cli_arg_provided("--link-surface-grid-u-size"):
        cfg.tacmap_link_surface_grid_u_size = float(link_surface["grid_u_size"])
    if not cli_arg_provided("--link-surface-grid-v-size"):
        cfg.tacmap_link_surface_grid_v_size = float(link_surface["grid_v_size"])
    if not cli_arg_provided("--link-surface-grid-center"):
        cfg.tacmap_link_surface_grid_center = tuple(float(value) for value in link_surface["grid_center"])


def parse_fingers_arg(fingers_text: str | None, *, default_finger: str) -> tuple[str, ...]:
    if fingers_text is None:
        return (str(default_finger),)
    raw_values = [value.strip() for value in str(fingers_text).split(",")]
    if len(raw_values) == 1 and raw_values[0].lower() == "all":
        return tuple(FINGER_CHOICES)

    out: list[str] = []
    invalid: list[str] = []
    for value in raw_values:
        if not value:
            continue
        if value not in FINGER_MAPS:
            invalid.append(value)
            continue
        if value not in out:
            out.append(value)
    if invalid:
        raise ValueError(f"--fingers contains unsupported finger(s): {invalid}; valid={tuple(FINGER_MAPS)}")
    if not out:
        raise ValueError("--fingers must name at least one finger")
    return tuple(out)


def resolve_focus_finger(active_fingers: tuple[str, ...], focus_finger: str | None) -> str:
    if not active_fingers:
        raise ValueError("At least one active finger is required")
    if focus_finger is None:
        return str(active_fingers[0])
    focus = str(focus_finger)
    if focus not in FINGER_MAPS:
        raise ValueError(f"--focus-finger must be one of {tuple(FINGER_MAPS)}")
    if focus not in active_fingers:
        raise ValueError(f"--focus-finger {focus!r} must also be listed in --fingers {active_fingers!r}")
    return focus


def apply_finger_cfg(cfg: IntegratedTactileEnvCfg, finger: str) -> None:
    spec = FINGER_MAPS[finger]
    cfg.press_touch_link = spec["touch_link"]
    cfg.press_points_npy = str(spec["points"])
    cfg.press_normals_npy = str(spec["normals"])
    cfg.press_tacmap_files_enabled = True
    cfg.press_touch_links = (str(spec["touch_link"]),)
    cfg.tactile_link_keywords = (str(spec["touch_link"]),)
    cfg.touch_collision_paths = (f"{spec['touch_link']}/collisions",)
    _apply_finger_link_surface_defaults(cfg, spec)


def apply_fingers_cfg(
    cfg: IntegratedTactileEnvCfg,
    fingers: tuple[str, ...],
    *,
    primary_finger: str | None = None,
) -> None:
    if not fingers:
        raise ValueError("At least one finger is required")
    primary = resolve_focus_finger(fingers, primary_finger)
    apply_finger_cfg(cfg, primary)

    touch_links: list[str] = []
    collision_paths: list[str] = []
    patch_specs: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
    for finger in fingers:
        spec = FINGER_MAPS[finger]
        link_name = str(spec["touch_link"])
        if link_name not in touch_links:
            touch_links.append(link_name)
            collision_paths.append(f"{link_name}/collisions")
        if finger == primary:
            continue
        center_l, normal_l, _, _ = load_legacy_finger_press_target(
            finger,
            correction_scale=float(cfg.press_map_correction_scale),
        )
        patch_specs.append((link_name, center_l, normal_l))

    cfg.press_touch_links = tuple(touch_links)
    cfg.tactile_link_keywords = tuple(touch_links)
    cfg.touch_collision_paths = tuple(collision_paths)
    cfg.press_link_patch_specs = tuple(patch_specs)


def load_legacy_finger_press_target(
    finger: str,
    *,
    correction_scale: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], str, str]:
    spec = FINGER_MAPS[finger]
    points_path = Path(spec["points"])
    normals_path = Path(spec["normals"])
    points = _load_tacmap_npy(points_path).astype(np.float64) * float(correction_scale)
    normals = _load_tacmap_npy(normals_path).astype(np.float64)
    valid = (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(normals).all(axis=-1)
        & (np.linalg.norm(points, axis=-1) > 1.0e-12)
        & (np.linalg.norm(normals, axis=-1) > 0.5)
    )
    if not np.any(valid):
        raise RuntimeError(f"No valid legacy tactile samples found for finger {finger!r}")
    pts = points[valid]
    nrms = normals[valid]
    nrms = nrms / (np.linalg.norm(nrms, axis=-1, keepdims=True) + 1.0e-12)
    center = np.mean(pts, axis=0).astype(np.float32)
    normal = _normalize_vector(np.mean(nrms, axis=0).astype(np.float32), fallback=np.array([0.0, 0.0, 1.0]))
    return (
        tuple(float(value) for value in center),
        tuple(float(value) for value in normal),
        str(points_path),
        str(normals_path),
    )


def pressure_pad_link_names_for_finger(finger: str) -> tuple[str, ...]:
    stem = PRESSURE_PAD_FINGER_STEMS.get(str(finger))
    if stem is None:
        raise ValueError(f"Unsupported pressure-pad finger {finger!r}; valid={tuple(PRESSURE_PAD_FINGER_STEMS)}")
    return tuple(f"right_{stem}{segment}_roll_touch_link" for segment in FOCUS_PRESSURE_PAD_SEGMENTS)


def focus_pressure_pad_specs_from_urdf(urdf_path: str | Path, finger: str) -> tuple[object, ...]:
    requested_links = pressure_pad_link_names_for_finger(finger)
    specs = load_pressure_pad_specs_from_urdf(urdf_path, require_files=False)
    specs_by_link = {str(spec.link_name): spec for spec in specs}
    selected = tuple(specs_by_link[link_name] for link_name in requested_links if link_name in specs_by_link)
    missing = [link_name for link_name in requested_links if link_name not in specs_by_link]
    if missing:
        print(
            f"[WARN] focus pressure-pad specs missing for finger={finger}: {missing} layout_urdf={urdf_path}",
            flush=True,
        )
    return selected


def all_pressure_pad_specs_from_urdf(urdf_path: str | Path) -> tuple[object, ...]:
    requested_links = tuple(link for finger in FINGER_CHOICES for link in pressure_pad_link_names_for_finger(finger))
    specs = load_pressure_pad_specs_from_urdf(urdf_path, require_files=False)
    specs_by_link = {str(spec.link_name): spec for spec in specs}
    selected = tuple(specs_by_link[link_name] for link_name in requested_links if link_name in specs_by_link)
    missing = [link_name for link_name in requested_links if link_name not in specs_by_link]
    if missing:
        print(
            f"[WARN] RL pressure-pad specs missing: {missing} layout_urdf={urdf_path}",
            flush=True,
        )
    return selected


def _pressure_pad_patch_spec(spec) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    taxel_map = spec.to_taxel_map()
    points_l = np.asarray(taxel_map.points_l, dtype=np.float32)
    normals_l = np.asarray(taxel_map.normals_l, dtype=np.float32)
    fallback_normal = np.zeros(3, dtype=np.float32)
    fallback_normal[int(spec.normal_axis)] = 1.0 if float(spec.normal_sign) >= 0.0 else -1.0
    center_l = tuple(float(value) for value in points_l.mean(axis=0))
    normal_l = tuple(float(value) for value in _normalize_vector(normals_l.mean(axis=0), fallback=fallback_normal))
    return str(spec.link_name), center_l, normal_l


def apply_pressure_pad_specs_cfg(
    cfg: IntegratedTactileEnvCfg,
    specs: tuple[object, ...],
    *,
    label: str,
    layout_urdf: str | Path,
) -> tuple[str, ...]:
    if not specs:
        return ()

    taxel_maps = [spec.to_taxel_map() for spec in specs]
    image_shapes = {tuple(int(value) for value in taxel_map.image_shape) for taxel_map in taxel_maps}
    if len(image_shapes) != 1:
        raise ValueError(f"{label} pressure pads must share one image shape, got {sorted(image_shapes)}")

    first_spec = specs[0]
    rows, cols = next(iter(image_shapes))
    cfg.num_rows = int(rows)
    cfg.num_cols = int(cols)
    cfg.row_distance = float(first_spec.row_distance)
    cfg.col_distance = float(first_spec.col_distance)
    cfg.point_distance = (
        float(first_spec.point_distance)
        if first_spec.point_distance is not None
        else float(min(float(first_spec.row_distance), float(first_spec.col_distance)))
    )
    cfg.normal_axis = int(first_spec.normal_axis)
    cfg.normal_offset = float(first_spec.normal_offset)
    cfg.normal_sign = float(first_spec.normal_sign)
    cfg.stiffness = float(first_spec.calibration.stiffness)
    cfg.damping = float(first_spec.calibration.damping)
    cfg.max_force = float(first_spec.calibration.max_force)
    cfg.pressure_gain = float(first_spec.calibration.gain)
    cfg.pressure_bias = float(first_spec.calibration.bias)
    cfg.pressure_gamma = float(first_spec.calibration.gamma)
    cfg.pressure_threshold = float(first_spec.calibration.threshold)
    cfg.taxel_area = float(first_spec.calibration.area)

    pressure_links = tuple(str(spec.link_name) for spec in specs)
    cfg.press_touch_links = tuple(dict.fromkeys((*cfg.press_touch_links, *pressure_links)))
    cfg.tactile_link_keywords = tuple(dict.fromkeys((*cfg.tactile_link_keywords, *pressure_links)))
    cfg.touch_collision_paths = tuple(
        dict.fromkeys((*cfg.touch_collision_paths, *(f"{link_name}/collisions" for link_name in pressure_links)))
    )
    cfg.press_link_patch_specs = tuple(
        dict.fromkeys((*cfg.press_link_patch_specs, *(_pressure_pad_patch_spec(spec) for spec in specs)))
    )
    print(
        f"[INFO] {label} pressure pads enabled: "
        f"links={pressure_links}, shape={rows}x{cols}, layout_urdf={layout_urdf}",
        flush=True,
    )
    return pressure_links


def apply_all_pressure_pad_cfg(
    cfg: IntegratedTactileEnvCfg,
    *,
    layout_urdf: str | Path,
) -> tuple[str, ...]:
    return apply_pressure_pad_specs_cfg(
        cfg,
        all_pressure_pad_specs_from_urdf(layout_urdf),
        label="RL all-finger",
        layout_urdf=layout_urdf,
    )


def apply_focus_pressure_pad_cfg(
    cfg: IntegratedTactileEnvCfg,
    *,
    focus_finger: str,
    layout_urdf: str | Path,
) -> tuple[str, ...]:
    return apply_pressure_pad_specs_cfg(
        cfg,
        focus_pressure_pad_specs_from_urdf(layout_urdf, focus_finger),
        label=f"focus finger={focus_finger}",
        layout_urdf=layout_urdf,
    )


def sensor_indices_for_link_names(labels: list[str], link_names: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    label_text = [str(label) for label in labels]
    for link_name in link_names:
        target = Path(str(link_name)).name.lower()
        matched_index = None
        for index, label in enumerate(label_text):
            label_lower = label.lower()
            if label_lower == target or target in label_lower or label_lower in target:
                matched_index = index
                break
        if matched_index is None:
            print(f"[WARN] focus pressure-pad display link missing from sensor labels: {link_name}", flush=True)
            continue
        if matched_index not in indices:
            indices.append(matched_index)
    return indices


def apply_pressure_layout_urdf_cfg(
    cfg: IntegratedTactileEnvCfg,
    urdf_path: str,
    *,
    link_name: str | None,
    origin_override_l: tuple[float, float, float] | None = None,
    normal_axis_override: int | None = None,
    normal_sign_override: float | None = None,
) -> UrdfPressurePadSpec:
    specs = load_pressure_pad_specs_from_urdf(urdf_path, require_files=False)
    spec = select_pressure_pad_spec(specs, link_name=link_name)
    if _is_fingertip_pressure_link(spec.link_name):
        raise ValueError(
            f"Pressure pad layout {spec.link_name!r} is on a fingertip; "
            "fingertips are visual tactile channels, not pressure pads."
        )
    if origin_override_l is not None:
        spec = replace(spec, origin_xyz=tuple(float(value) for value in origin_override_l))
    if normal_axis_override is not None:
        spec = replace(spec, normal_axis=int(normal_axis_override))
    if normal_sign_override is not None:
        spec = replace(spec, normal_sign=float(normal_sign_override))
    if spec.origin_semantics != "pad_surface":
        raise ValueError(
            "Integrated WarpSDF pressure layout currently requires pad_surface origins; "
            f"got {spec.origin_semantics!r} for {spec.link_name!r}."
        )
    if spec.points_npy is not None or spec.normals_npy is not None:
        raise ValueError(
            "Integrated WarpSDF pressure layout currently supports grid pads only; "
            "use rows/cols with point_distance, row/col pitch, or pad_size; inspect point-map files separately."
        )
    taxel_map = spec.to_taxel_map()
    rows, cols = (int(value) for value in taxel_map.image_shape)
    points_l = np.asarray(taxel_map.points_l, dtype=np.float32)
    normals_l = np.asarray(taxel_map.normals_l, dtype=np.float32)
    fallback_normal = np.zeros(3, dtype=np.float32)
    fallback_normal[int(spec.normal_axis)] = 1.0 if float(spec.normal_sign) >= 0.0 else -1.0

    cfg.press_touch_link = spec.link_name
    cfg.tactile_link_keywords = (spec.link_name,)
    cfg.touch_collision_paths = (f"{spec.link_name}/collisions",)
    cfg.press_points_npy = ""
    cfg.press_normals_npy = ""
    cfg.press_tacmap_files_enabled = False
    cfg.press_center_l = tuple(float(value) for value in points_l.mean(axis=0))
    cfg.press_normal_l = tuple(float(value) for value in _normalize_vector(normals_l.mean(axis=0), fallback=fallback_normal))
    cfg.press_patch_pos_l = tuple(float(value) for value in spec.origin_xyz)
    cfg.press_patch_quat_l = rpy_to_quat_wxyz(spec.origin_rpy)
    cfg.num_rows = rows
    cfg.num_cols = cols
    cfg.row_distance = float(spec.row_distance)
    cfg.col_distance = float(spec.col_distance)
    cfg.point_distance = (
        float(spec.point_distance)
        if spec.point_distance is not None
        else float(min(float(spec.row_distance), float(spec.col_distance)))
    )
    cfg.normal_axis = int(spec.normal_axis)
    cfg.normal_offset = float(spec.normal_offset)
    cfg.normal_sign = float(spec.normal_sign)
    cfg.stiffness = float(spec.calibration.stiffness)
    cfg.damping = float(spec.calibration.damping)
    cfg.max_force = float(spec.calibration.max_force)
    cfg.pressure_gain = float(spec.calibration.gain)
    cfg.pressure_bias = float(spec.calibration.bias)
    cfg.pressure_gamma = float(spec.calibration.gamma)
    cfg.pressure_threshold = float(spec.calibration.threshold)
    cfg.taxel_area = float(spec.calibration.area)
    return spec


def apply_presser_cfg(
    cfg: IntegratedTactileEnvCfg,
    presser: str,
    *,
    extra_rot_axis: str,
    extra_rot_deg: float | None,
) -> tuple[str, float]:
    spec = PRESSER_SPECS[presser]
    usd_path = Path(spec["usd"])
    if not usd_path.is_file():
        raise FileNotFoundError(f"Presser USD not found: {usd_path}")
    cfg.press_object_usd_path = str(usd_path)

    if extra_rot_axis == "auto":
        extra_rot_axis = str(spec["default_extra_rot_axis"])
    if extra_rot_deg is None:
        extra_rot_deg = float(spec["default_extra_rot_deg"])

    base_rot = tuple(float(v) for v in spec["rot"])
    extra_rot = axis_angle_to_quat(extra_rot_axis, float(extra_rot_deg))
    cfg.press_object_rot_in_touch_frame = quat_mul(base_rot, extra_rot)
    return extra_rot_axis, float(extra_rot_deg)


def presser_flip_axis(presser: str, requested_flip_axis: str) -> str:
    if requested_flip_axis != "auto":
        return requested_flip_axis
    return str(PRESSER_SPECS[presser]["default_flip_axis"])


def main():
    pressure_pad_run = args.pressure_layout_urdf is not None
    legacy_integrated_run = not pressure_pad_run
    pressure_contact_model = args.pressure_contact_model or (
        "surface_gap" if pressure_pad_run else "signed_penetration"
    )
    if args.pressure_contact_model is None and bool(args.mesh_unsigned_shell_as_contact):
        pressure_contact_model = "surface_gap"
    active_fingers = parse_fingers_arg(args.fingers, default_finger=args.finger)
    if pressure_pad_run and (args.fingers is not None or args.focus_finger is not None):
        raise ValueError(
            "--fingers/--focus-finger are for the Revo21 visual-tactile path and cannot be combined with "
            "--pressure-layout-urdf"
        )
    primary_finger = resolve_focus_finger(active_fingers, args.focus_finger)
    if pressure_pad_run and (
        args.enable_fots
        or args.enable_tacex_rgb
        or args.enable_tacsl_shear
        or args.enable_hydroshear_marker
        or args.show_tacmap_points
        or args.show_tacmap_active_points
        or args.show_tacmap_rays
        or args.show_tacmap_link_surfaces
        or args.enable_normal_ray_pressure
        or args.enable_geometry_normal_ray_pressure
        or args.pressure_view_source != "warpsdf"
    ):
        raise ValueError(
            "TacMap/FOTS/TacEX/TacSL/HydroShear and alternate pressure views are fingertip/diagnostic channels; "
            "pressure-pad URDF runs should use the WarpSDF pressure map only."
        )
    cfg = IntegratedTactileEnvCfg(
        headless=bool(getattr(args, "headless", True)),
        device=str(getattr(args, "device", "cuda:0")),
        enable_camera=bool(getattr(args, "enable_cameras", False)),
    )
    if args.set_viewport_camera is None:
        cfg.set_viewport_camera = not bool(cfg.headless)
    else:
        cfg.set_viewport_camera = bool(args.set_viewport_camera)
    if args.viewport_camera_eye is not None:
        cfg.viewport_camera_eye = tuple(float(v) for v in args.viewport_camera_eye)
    if args.viewport_camera_target is not None:
        cfg.viewport_camera_target = tuple(float(v) for v in args.viewport_camera_target)
    robot_urdf_path = (
        Path(args.robot_urdf).expanduser()
        if args.robot_urdf is not None
        else Path(cfg.urdf_path)
    )
    cfg.urdf_path = str(robot_urdf_path)
    cfg.usd_output_dir = str(
        Path(args.robot_usd_output_dir).expanduser()
        if args.robot_usd_output_dir is not None
        else default_robot_usd_output_dir(robot_urdf_path)
    )
    cfg.enable_press_motion = args.mode == "press"
    cfg.enable_sample_point_view = args.mode == "points"
    cfg.enable_tacmap = not bool(args.disable_tacmap or pressure_pad_run)
    cfg.tacmap_resolution_step = max(1, int(args.tacmap_resolution_step))
    if legacy_integrated_run:
        if not cli_arg_provided("--tacmap-max-distance"):
            args.tacmap_max_distance = 0.015
        if not cli_arg_provided("--tacmap-ray-mode"):
            args.tacmap_ray_mode = "link_surface"
    cfg.tacmap_max_distance = float(args.tacmap_max_distance)
    cfg.tacmap_invert_normals_file = bool(args.tacmap_invert_normals) and not bool(args.tacmap_use_original_normals)
    cfg.tacmap_debug_hit_stats = bool(args.tacmap_debug_hit_stats)
    cfg.tacmap_debug_hit_stats_every = int(args.tacmap_debug_hit_every)
    cfg.tacmap_debug_vis = bool(args.show_tacmap_points or args.show_tacmap_active_points)
    cfg.tacmap_debug_viz_active_points = bool(args.show_tacmap_active_points)
    cfg.tacmap_debug_viz_active_threshold = float(args.tacmap_active_point_threshold)
    cfg.tacmap_ray_mode = str(args.tacmap_ray_mode)
    cfg.tacmap_link_surface_width = 240
    cfg.tacmap_link_surface_height = 240
    cfg.tacmap_link_surface_ray_axis = str(args.link_surface_ray_axis)
    cfg.tacmap_link_surface_ray_direction = (
        tuple(float(v) for v in args.link_surface_ray_direction)
        if args.link_surface_ray_direction is not None
        else None
    )
    cfg.tacmap_link_surface_use_mean_normal = not bool(args.link_surface_use_axis_direction)
    cfg.tacmap_link_surface_grid_u_axis = str(args.link_surface_grid_u_axis)
    cfg.tacmap_link_surface_grid_v_axis = str(args.link_surface_grid_v_axis)
    cfg.tacmap_link_surface_grid_u_size = float(args.link_surface_grid_u_size)
    cfg.tacmap_link_surface_grid_v_size = float(args.link_surface_grid_v_size)
    cfg.tacmap_link_surface_grid_center = tuple(float(v) for v in args.link_surface_grid_center)
    cfg.tacmap_link_surface_debug_surfaces = bool(args.show_tacmap_link_surfaces)
    cfg.tacmap_link_surface_debug_rays = bool(args.show_tacmap_rays)
    cfg.tacmap_link_surface_debug_ray_length = float(args.link_surface_ray_viz_length)
    cfg.tacmap_link_surface_debug_ray_width = float(args.link_surface_ray_viz_width)
    cfg.enable_tacsl_force_field = bool(args.enable_tacsl_shear)
    cfg.tacsl_sdf_gradient_eps = float(args.tacsl_sdf_gradient_eps)
    cfg.tacsl_normal_contact_stiffness = float(args.tacsl_normal_contact_stiffness)
    cfg.tacsl_tangential_stiffness = float(args.tacsl_tangential_stiffness)
    cfg.tacsl_friction_coefficient = float(args.tacsl_friction_coefficient)
    extra_rot_axis, extra_rot_deg = apply_presser_cfg(
        cfg,
        args.presser,
        extra_rot_axis=str(args.presser_extra_rot_axis),
        extra_rot_deg=args.presser_extra_rot_deg,
    )
    selected_pressure_pad_spec: UrdfPressurePadSpec | None = None
    focus_pressure_pad_links: tuple[str, ...] = ()
    press_motion_source = str(args.press_motion_source or ("pressure_layout" if pressure_pad_run else "legacy_finger_map"))
    press_motion_source_metadata: dict[str, object] = {"source": press_motion_source}
    if args.pressure_layout_link is not None and args.pressure_layout_urdf is None:
        raise ValueError("--pressure-layout-link requires --pressure-layout-urdf")
    if args.pressure_layout_urdf is None:
        if args.robot_urdf is not None:
            raise ValueError(
                "--pressure-layout-urdf is required with --robot-urdf for pressure taxel maps; "
                "omit --robot-urdf to run the legacy TacMap finger demo."
            )
        apply_fingers_cfg(cfg, active_fingers, primary_finger=primary_finger)
        all_pressure_pad_links = apply_all_pressure_pad_cfg(
            cfg,
            layout_urdf=DEFAULT_PRESSURE_PAD_LAYOUT_URDF,
        )
        focus_pressure_pad_links = tuple(
            link_name for link_name in pressure_pad_link_names_for_finger(primary_finger) if link_name in all_pressure_pad_links
        )
    if args.pressure_layout_urdf is not None:
        selected_pressure_pad_spec = apply_pressure_layout_urdf_cfg(
            cfg,
            str(args.pressure_layout_urdf),
            link_name=args.pressure_layout_link,
            origin_override_l=(
                tuple(float(value) for value in args.pressure_layout_origin_l)
                if args.pressure_layout_origin_l is not None
                else None
            ),
            normal_axis_override=(int(args.normal_axis) if args.normal_axis is not None else None),
            normal_sign_override=(float(args.normal_sign) if args.normal_sign is not None else None),
        )
    if press_motion_source == "legacy_finger_map":
        legacy_center_l, legacy_normal_l, legacy_points, legacy_normals = load_legacy_finger_press_target(
            primary_finger,
            correction_scale=float(cfg.press_map_correction_scale),
        )
        cfg.press_center_l = legacy_center_l
        cfg.press_normal_l = legacy_normal_l
        press_motion_source_metadata.update(
            {
                "legacy_finger": str(primary_finger),
                "primary_finger": str(primary_finger),
                "active_fingers": [str(value) for value in active_fingers],
                "legacy_points_npy": legacy_points,
                "legacy_normals_npy": legacy_normals,
                "legacy_point_correction_scale": float(cfg.press_map_correction_scale),
            }
        )
        print(
            "[INFO] press motion source=legacy_finger_map: "
            f"focus_finger={primary_finger}, active_fingers={active_fingers}, "
            f"center_l={legacy_center_l}, normal_l={legacy_normal_l}; "
            f"press_link={cfg.press_touch_link}, sensor_links={cfg.press_touch_links}",
            flush=True,
        )

    if args.num_rows is not None:
        cfg.num_rows = int(args.num_rows)
    if args.num_cols is not None:
        cfg.num_cols = int(args.num_cols)
    if args.point_distance is not None:
        cfg.point_distance = float(args.point_distance)
        cfg.row_distance = float(args.point_distance)
        cfg.col_distance = float(args.point_distance)
    if args.row_distance is not None:
        cfg.row_distance = float(args.row_distance)
    if args.col_distance is not None:
        cfg.col_distance = float(args.col_distance)
    if args.normal_offset is not None:
        cfg.normal_offset = float(args.normal_offset)
    if args.normal_axis is not None and selected_pressure_pad_spec is None:
        cfg.normal_axis = int(args.normal_axis)
    if args.normal_sign is not None and selected_pressure_pad_spec is None:
        cfg.normal_sign = float(args.normal_sign)
    pressure_calib_overrides = load_pressure_calibration(
        args.pressure_calib,
        image_shape=(int(cfg.num_rows), int(cfg.num_cols)),
    )
    pressure_cli_options = {
        "stiffness": "--pressure-stiffness",
        "damping": "--pressure-damping",
        "max_force": "--pressure-max-force",
        "gain": "--pressure-gain",
        "bias": "--pressure-bias",
        "gamma": "--pressure-gamma",
        "threshold": "--pressure-threshold",
        "taxel_area": "--taxel-area",
    }
    apply_pressure_calibration_overrides(
        cfg,
        {
            key: value
            for key, value in pressure_calib_overrides.items()
            if not cli_arg_provided(pressure_cli_options[key])
        },
    )
    if args.pressure_stiffness is not None:
        cfg.stiffness = float(args.pressure_stiffness)
    if args.pressure_damping is not None:
        cfg.damping = float(args.pressure_damping)
    if args.pressure_max_force is not None:
        cfg.max_force = float(args.pressure_max_force)
    if cli_arg_provided("--pressure-gain"):
        cfg.pressure_gain = float(args.pressure_gain)
    if cli_arg_provided("--pressure-bias"):
        cfg.pressure_bias = float(args.pressure_bias)
    if cli_arg_provided("--pressure-gamma"):
        cfg.pressure_gamma = float(args.pressure_gamma)
    if cli_arg_provided("--pressure-threshold"):
        cfg.pressure_threshold = float(args.pressure_threshold)
    if cli_arg_provided("--taxel-area"):
        cfg.taxel_area = float(args.taxel_area)
    cfg.penetration_deadband = max(0.0, float(args.penetration_deadband))
    if args.mesh_shell_thickness is not None:
        if float(args.mesh_shell_thickness) < 0.0:
            raise ValueError("--mesh-shell-thickness must be non-negative.")
        cfg.mesh_shell_thickness = float(args.mesh_shell_thickness)
    cfg.mesh_unsigned_contact_mode = str(args.pressure_surface_gap_mode)
    cfg.pressure_response_model = str(args.pressure_response_model)
    cfg.mesh_signed = pressure_contact_model == "signed_penetration"
    cfg.mesh_unsigned_shell_as_contact = pressure_contact_model == "surface_gap"
    cfg.enable_physx_contact_force_map = bool(args.enable_physx_contact_map or args.contact_benchmark_out) and not bool(
        args.disable_physx_contact_map
    )
    cfg.physx_contact_max_data_count_per_prim = max(1, int(args.physx_contact_max_data))
    cfg.physx_contact_kernel_sigma = (
        float(args.physx_contact_kernel_sigma) if args.physx_contact_kernel_sigma is not None else None
    )
    cfg.physx_contact_kernel_radius = (
        float(args.physx_contact_kernel_radius) if args.physx_contact_kernel_radius is not None else None
    )
    cfg.gate_warpsdf_with_physx_contact = (
        bool(args.enable_warpsdf_contact_gate)
        and bool(cfg.enable_physx_contact_force_map)
        and not bool(args.disable_warpsdf_contact_gate)
    )
    cfg.warpsdf_contact_gate_min_contacts = max(1, int(args.warpsdf_contact_gate_min_contacts))

    cfg.debug_vis = bool(args.show_sample_points or args.show_sample_axes or cfg.enable_sample_point_view)
    cfg.debug_vis_show_all_taxels = bool(args.show_sample_points or cfg.enable_sample_point_view)
    cfg.debug_vis_show_axes = bool(args.show_sample_axes)
    cfg.debug_vis_point_radius = float(args.sample_point_radius)
    cfg.show_pressure_pad_centers = bool(args.show_pressure_pad_centers)
    cfg.pressure_pad_center_marker_radius = max(1.0e-6, float(args.pressure_pad_center_radius))
    cfg.show_pressure_pad_taxel_points = bool(
        (legacy_integrated_run or args.show_pressure_pad_taxel_points) and not args.hide_pressure_pad_taxel_points
    )
    cfg.pressure_pad_taxel_point_radius = max(1.0e-6, float(args.pressure_pad_taxel_point_radius))
    cfg.pressure_pad_taxel_layout_urdf = str(
        Path(args.pressure_layout_urdf).expanduser()
        if args.pressure_layout_urdf is not None
        else DEFAULT_PRESSURE_PAD_LAYOUT_URDF
    )
    cfg.press_motion_frame = str(args.press_motion_frame or ("touch" if pressure_pad_run else cfg.press_motion_frame))
    cfg.press_start_offset = float(args.press_start_offset)
    cfg.press_end_offset = float(args.press_end_offset)
    if args.press_distance is not None:
        if cli_arg_provided("--press-end-offset"):
            raise ValueError("--press-distance sets --press-end-offset; pass only one of them.")
        distance = float(args.press_distance)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("--press-distance must be a finite non-negative distance in meters.")
        cfg.press_end_offset = float(cfg.press_start_offset) - distance
    cfg.press_indent_depth = None if args.press_indent_depth is None else float(args.press_indent_depth)
    if cfg.press_indent_depth is not None and (not math.isfinite(cfg.press_indent_depth) or cfg.press_indent_depth < 0.0):
        raise ValueError("--press-indent-depth must be a finite non-negative distance in meters.")
    cfg.press_contact_search_distance = (
        None if args.press_contact_search_distance is None else float(args.press_contact_search_distance)
    )
    if cfg.press_contact_search_distance is not None and (
        not math.isfinite(cfg.press_contact_search_distance) or cfg.press_contact_search_distance < 0.0
    ):
        raise ValueError("--press-contact-search-distance must be a finite non-negative distance in meters.")
    cfg.press_contact_threshold = float(args.press_contact_threshold)
    if not math.isfinite(cfg.press_contact_threshold) or cfg.press_contact_threshold < 0.0:
        raise ValueError("--press-contact-threshold must be a finite non-negative distance in meters.")
    cfg.press_steps = int(args.press_steps)
    cfg.press_slide_distance = float(args.press_slide_distance)
    cfg.press_slide_steps = int(args.press_slide_steps)
    cfg.press_slide_axis_l = axis_to_vector(
        str(args.press_slide_axis),
        grid_u_axis=cfg.tacmap_link_surface_grid_u_axis,
        grid_v_axis=cfg.tacmap_link_surface_grid_v_axis,
        ray_axis=cfg.tacmap_link_surface_ray_axis,
        ray_direction=cfg.tacmap_link_surface_ray_direction,
    )
    cfg.plug_scale = max(1.0e-6, float(args.presser_scale))
    cfg.press_object_contact_offset = float(args.presser_contact_offset)
    cfg.press_object_rest_offset = float(args.presser_rest_offset)
    cfg.press_object_mass = max(1.0e-9, float(args.presser_mass))
    cfg.press_object_control = str(args.press_object_control)
    enable_focus_pressure_pad_presser = (
        bool(args.enable_focus_pressure_pad_presser)
        and not bool(args.disable_focus_pressure_pad_presser)
    )
    if focus_pressure_pad_links and enable_focus_pressure_pad_presser:
        segment = str(args.focus_pressure_pad_presser_segment)
        target_link = next(
            (link for link in focus_pressure_pad_links if f"{segment}_roll_touch_link" in str(link)),
            focus_pressure_pad_links[0],
        )
        pressure_pad_presser_scale = (
            float(args.focus_pressure_pad_presser_scale)
            if args.focus_pressure_pad_presser_scale is not None
            else float(cfg.plug_scale)
        )
        if not math.isfinite(pressure_pad_presser_scale) or pressure_pad_presser_scale <= 0.0:
            raise ValueError("--focus-pressure-pad-presser-scale must be a positive finite value.")
        pressure_pad_press_start_offset = (
            float(args.pressure_pad_press_start_offset)
            if args.pressure_pad_press_start_offset is not None
            else float(args.focus_pressure_pad_presser_start_offset)
        )
        pressure_pad_press_end_offset = (
            float(args.pressure_pad_press_end_offset)
            if args.pressure_pad_press_end_offset is not None
            else float(args.focus_pressure_pad_presser_end_offset)
        )
        for option_name, offset_value in (
            ("--pressure-pad-press-start-offset", pressure_pad_press_start_offset),
            ("--pressure-pad-press-end-offset", pressure_pad_press_end_offset),
        ):
            if not math.isfinite(float(offset_value)):
                raise ValueError(f"{option_name} must be finite.")
        pressure_pad_press_steps = (
            int(args.pressure_pad_press_steps)
            if args.pressure_pad_press_steps is not None
            else None
        )
        if pressure_pad_press_steps is not None and pressure_pad_press_steps <= 0:
            raise ValueError("--pressure-pad-press-steps must be a positive integer.")
        pressure_pad_presser_offset_l = tuple(float(value) for value in args.focus_pressure_pad_presser_offset_l)
        if len(pressure_pad_presser_offset_l) != 3 or not all(
            math.isfinite(value) for value in pressure_pad_presser_offset_l
        ):
            raise ValueError("--focus-pressure-pad-presser-offset-l must be three finite values.")

        def _optional_presser_offset_l(option_name: str, values) -> tuple[float, float, float] | None:
            if values is None:
                return None
            offset_l = tuple(float(value) for value in values)
            if len(offset_l) != 3 or not all(math.isfinite(value) for value in offset_l):
                raise ValueError(f"{option_name} must be three finite values.")
            return offset_l

        pressure_pad_presser_start_offset_l = _optional_presser_offset_l(
            "--focus-pressure-pad-presser-start-offset-l",
            args.focus_pressure_pad_presser_start_offset_l,
        )
        pressure_pad_presser_end_offset_l = _optional_presser_offset_l(
            "--focus-pressure-pad-presser-end-offset-l",
            args.focus_pressure_pad_presser_end_offset_l,
        )
        cfg.enable_pressure_pad_presser = True
        cfg.pressure_pad_presser_link = str(target_link)
        cfg.pressure_pad_presser_usd_path = str(cfg.press_object_usd_path)
        cfg.pressure_pad_presser_scale = pressure_pad_presser_scale
        cfg.pressure_pad_presser_start_offset = pressure_pad_press_start_offset
        cfg.pressure_pad_presser_end_offset = pressure_pad_press_end_offset
        cfg.pressure_pad_presser_steps = pressure_pad_press_steps
        cfg.pressure_pad_presser_offset_l = pressure_pad_presser_offset_l
        cfg.pressure_pad_presser_start_offset_l = pressure_pad_presser_start_offset_l
        cfg.pressure_pad_presser_end_offset_l = pressure_pad_presser_end_offset_l
        print(
            "[INFO] focus pressure-pad presser enabled: "
            f"segment={segment}, link={cfg.pressure_pad_presser_link}, "
            f"offset={cfg.pressure_pad_presser_start_offset:g}->{cfg.pressure_pad_presser_end_offset:g}, "
            f"steps={cfg.pressure_pad_presser_steps or cfg.press_steps}, "
            f"offset_l={cfg.pressure_pad_presser_offset_l}, "
            f"start_offset_l={cfg.pressure_pad_presser_start_offset_l}, "
            f"end_offset_l={cfg.pressure_pad_presser_end_offset_l}, "
            f"scale={cfg.pressure_pad_presser_scale:g}, usd={cfg.pressure_pad_presser_usd_path}",
            flush=True,
        )
    else:
        cfg.enable_pressure_pad_presser = False
    robot_world_pos = _optional_vec3(args.robot_world_pos, field_name="--robot-world-pos")
    robot_world_quat = _optional_unit_quat_wxyz(args.robot_world_quat_wxyz, field_name="--robot-world-quat-wxyz")
    if robot_world_pos is not None:
        cfg.robot_init_pos = robot_world_pos
    if robot_world_quat is not None:
        cfg.robot_init_rot_wxyz = robot_world_quat
    cfg.press_setup_edit_robot_pose = bool(args.setup_edit_robot_pose)
    if args.plug_fix_base is not None:
        cfg.plug_fix_base = bool(args.plug_fix_base)
    cfg.press_motion_actor = str(args.press_motion_actor)
    if str(args.presser) == "ball_probe" and cfg.press_motion_actor == "object":
        cfg.press_hand_axis_link = str(
            args.press_hand_axis_link or ("" if pressure_pad_run else "touch_nearest_world_z")
        )
        axis_text = str(args.press_hand_axis) if cli_arg_provided("--press-hand-axis") else "-z"
    elif str(args.presser) == "ball_probe" and cfg.press_motion_actor == "hand":
        cfg.press_hand_axis_link = str(args.press_hand_axis_link or "world")
        axis_text = str(args.press_hand_axis) if cli_arg_provided("--press-hand-axis") else "-z"
    else:
        cfg.press_hand_axis_link = str(args.press_hand_axis_link or "")
        axis_text = str(args.press_hand_axis)
    cfg.press_hand_axis_l = axis_to_vector(axis_text)
    default_presser_collision = (not bool(pressure_pad_run)) or bool(cfg.enable_physx_contact_force_map)
    cfg.press_object_collision_enabled = (
        bool(default_presser_collision)
        if args.presser_collision_enabled is None
        else bool(args.presser_collision_enabled)
    )
    if bool(cfg.enable_physx_contact_force_map) and not bool(cfg.press_object_collision_enabled):
        raise ValueError(
            "PhysX contact-map diagnostics require presser collision. "
            "Pass --enable-presser-collision, or disable the PhysX contact map."
        )
    if cfg.press_motion_actor == "hand" or cfg.press_setup_edit_robot_pose:
        cfg.fix_base = False
    cfg.press_finger_joints = tuple(args.press_finger_joint or ("right_midmcp_roll_joint",))
    cfg.press_finger_start_rad = float(args.press_finger_start_rad)
    cfg.press_finger_end_rad = float(args.press_finger_end_rad)
    cfg.press_finger_kinematic_sensor_pose = not bool(args.disable_press_finger_kinematic_sensor_pose)
    cfg.press_hold_joint_pose = str(args.press_hold_joint_pose)
    initial_joint_degrees = args.press_initial_joint_deg
    parsed_initial_joint_degrees = []
    for joint_name, degrees_text in initial_joint_degrees or ():
        degrees = float(degrees_text)
        if not math.isfinite(degrees):
            raise ValueError(f"--press-initial-joint-deg {joint_name} must use a finite degree value.")
        parsed_initial_joint_degrees.append((str(joint_name), degrees))
    cfg.press_initial_joint_degrees = tuple(parsed_initial_joint_degrees)
    cfg.lock_press_finger_joints = bool(args.lock_press_finger_joints)
    presser_world_pos = _optional_vec3(args.presser_world_pos, field_name="--presser-world-pos")
    presser_world_quat_wxyz = _optional_unit_quat_wxyz(
        args.presser_world_quat_wxyz,
        field_name="--presser-world-quat-wxyz",
    )
    if presser_world_quat_wxyz is None and str(args.presser) == "ball_probe":
        presser_world_quat_wxyz = BALL_PROBE_PRESSURE_PAD_QUAT_WXYZ
    if presser_world_pos is not None or presser_world_quat_wxyz is not None:
        current_pose = tuple(float(value) for value in cfg.plug_default_pose)
        pos_xyz = presser_world_pos or current_pose[:3]
        if presser_world_quat_wxyz is None:
            quat_xyzw = current_pose[3:7]
        else:
            qw, qx, qy, qz = presser_world_quat_wxyz
            quat_xyzw = (qx, qy, qz, qw)
        cfg.plug_default_pose = tuple(float(value) for value in (*pos_xyz, *quat_xyzw))
    cfg.enable_touch_compliant_material = not bool(args.disable_touch_compliant_material)
    cfg.compliant_contact_stiffness = float(args.touch_stiffness)
    cfg.compliant_contact_damping = float(args.touch_damping)
    cfg.touch_contact_offset = float(args.touch_contact_offset)
    cfg.touch_rest_offset = float(args.touch_rest_offset)
    flip_axis = presser_flip_axis(args.presser, args.flip_axis)
    cfg.press_object_flip_quat_in_touch_frame = flip_axis_to_quat(flip_axis)
    apply_press_motion_cli_overrides(cfg)

    live_control = (
        PressLiveControl(
            args.live_control_file,
            cfg,
            write_template=bool(args.write_live_control_template),
        )
        if args.live_control_file is not None
        else None
    )
    if live_control is not None:
        print(f"[INFO] live press control file: {live_control.path}", flush=True)
    presser_pose_recorder = (
        PresserPoseRecorder(
            args.presser_pose_save_file,
            cfg,
            presser=str(args.presser),
            save_every=int(args.presser_pose_save_every),
        )
        if args.presser_pose_save_file is not None
        else None
    )
    if presser_pose_recorder is not None:
        print(f"[INFO] presser pose save file: {presser_pose_recorder.path}", flush=True)

    hand_axis_l_text = tuple(float(v) for v in cfg.press_hand_axis_l)
    effective_pressure_response_model = str(cfg.pressure_response_model)
    if effective_pressure_response_model == "auto":
        effective_pressure_response_model = "gap_fraction" if pressure_contact_model == "surface_gap" else "penetration_kv"
    pressure_kv_usage = "unused" if effective_pressure_response_model == "gap_fraction" else "active"
    print(
        f"[INFO] integrated mode={args.mode}, legacy_finger_arg={args.finger}, "
        f"active_fingers={active_fingers}, focus_finger={primary_finger}, presser={args.presser}, "
        f"press_touch_link={cfg.press_touch_link}, "
        f"flip_axis={flip_axis}, extra_rot={extra_rot_deg:g}deg/{extra_rot_axis}, "
        f"press_motion_source={press_motion_source}, "
        f"press_object_control={cfg.press_object_control}, press_motion_actor={cfg.press_motion_actor}, "
        f"press_hand_axis={cfg.press_hand_axis_link or '<sensor_normal>'}/{hand_axis_l_text}, "
        f"press_finger_joints={list(cfg.press_finger_joints)}, "
        f"press_finger_end={cfg.press_finger_end_rad:g}, "
        f"press_finger_kinematic_sensor_pose={cfg.press_finger_kinematic_sensor_pose}, "
        f"robot_usd_dir={cfg.usd_output_dir}, "
        f"offset={cfg.press_start_offset}->{cfg.press_end_offset} "
        f"(distance={cfg.press_start_offset - cfg.press_end_offset:g}m), steps={cfg.press_steps}, "
        f"indent_depth={cfg.press_indent_depth}, contact_search={cfg.press_contact_search_distance}, "
        f"contact_threshold={cfg.press_contact_threshold:g}, "
        f"press_frame={cfg.press_motion_frame}, "
        f"slide={cfg.press_slide_distance:g}m/{cfg.press_slide_steps}steps/{args.press_slide_axis}"
        f"->axis_l={cfg.press_slide_axis_l}, "
        f"pressure_k={calibration_value_text(cfg.stiffness)}({pressure_kv_usage}), "
        f"pressure_c={calibration_value_text(cfg.damping)}({pressure_kv_usage}), "
        f"pressure_max={calibration_value_text(cfg.max_force)}, "
        f"pressure_gain={calibration_value_text(cfg.pressure_gain)}, "
        f"pressure_bias={calibration_value_text(cfg.pressure_bias)}, "
        f"pressure_gamma={calibration_value_text(cfg.pressure_gamma)}, "
        f"pressure_threshold={calibration_value_text(cfg.pressure_threshold)}, "
        f"pressure_calib={args.pressure_calib or 'inline'}, "
        f"pressure_contact_model={pressure_contact_model}, surface_gap_mode={cfg.mesh_unsigned_contact_mode}, "
        f"response_model={cfg.pressure_response_model}->{effective_pressure_response_model}, "
        f"shell={cfg.mesh_shell_thickness:g}m, "
        f"physx_contact_map={'on' if cfg.enable_physx_contact_force_map else 'off'}, "
        f"warpsdf_contact_gate={'on' if cfg.gate_warpsdf_with_physx_contact else 'off'}, "
        f"penetration_deadband={cfg.penetration_deadband:g}, "
        f"presser_scale={cfg.plug_scale:g}, presser_contact={cfg.press_object_contact_offset:g}/{cfg.press_object_rest_offset:g}, "
        f"presser_collision={'on' if cfg.press_object_collision_enabled else 'off'}, "
        f"touch_compliance={'on' if cfg.enable_touch_compliant_material else 'off'} "
        f"k={cfg.compliant_contact_stiffness:g} d={cfg.compliant_contact_damping:g} "
        f"touch_offsets={cfg.touch_contact_offset:g}/{cfg.touch_rest_offset:g}, "
        f"tacmap={'on' if cfg.enable_tacmap else 'off'}, tacmap_ray_mode={cfg.tacmap_ray_mode}",
        flush=True,
    )
    if cfg.tacmap_ray_mode == "link_surface":
        ray_source = (
            f"custom={cfg.tacmap_link_surface_ray_direction}"
            if cfg.tacmap_link_surface_ray_direction is not None
            else ("axis=" + cfg.tacmap_link_surface_ray_axis if args.link_surface_use_axis_direction else "mean tactile normal")
        )
        print(
            f"[INFO] link_surface rays: dir={ray_source}, "
            f"u={cfg.tacmap_link_surface_grid_u_axis}/{cfg.tacmap_link_surface_grid_u_size:g}m, "
            f"v={cfg.tacmap_link_surface_grid_v_axis}/{cfg.tacmap_link_surface_grid_v_size:g}m, "
            f"center={cfg.tacmap_link_surface_grid_center}, "
            f"max_dist={cfg.tacmap_max_distance:g}m, "
            f"surfaces_viz={'on' if cfg.tacmap_link_surface_debug_surfaces else 'off'}, "
            f"rays_viz={'on' if cfg.tacmap_link_surface_debug_rays else 'off'}",
            flush=True,
        )

    env = IntegratedTactileEnv(cfg, simulation_app=simulation_app)
    obs, _ = env.reset()
    label_source = obs.get("pressure_force_map", obs.get("tactile"))
    label_count = int(label_source.shape[0]) if label_source is not None else int(getattr(env, "_tactile_sensor_count", 0))
    labels = pressure_sensor_labels_from_env(env, label_count, fallback_label=primary_finger)
    print(f"[INFO] tactile sensor labels: {labels}", flush=True)
    display_finger = primary_finger
    display_sensor_index = active_fingers.index(display_finger) if display_finger in active_fingers else 0
    if label_count > 0:
        display_sensor_index = max(0, min(int(display_sensor_index), label_count - 1))
    display_label = labels[display_sensor_index] if display_sensor_index < len(labels) else display_finger
    focus_pressure_pad_sensor_indices = sensor_indices_for_link_names(labels, focus_pressure_pad_links)
    print(
        f"[INFO] integrated panes display sensor: index={display_sensor_index}, "
        f"finger={display_finger}, label={display_label}; 3D debug marker points still use all active fingers",
        flush=True,
    )
    if focus_pressure_pad_sensor_indices:
        focus_pressure_pad_labels = [
            labels[index] if 0 <= index < len(labels) else f"S{index}" for index in focus_pressure_pad_sensor_indices
        ]
        print(
            f"[INFO] focus pressure-pad panes: indices={focus_pressure_pad_sensor_indices}, "
            f"labels={focus_pressure_pad_labels}",
            flush=True,
        )
    if args.print_rubber_link_poses:
        print_rubber_link_poses(env)
    press_debug_logger = (
        PressDebugLogger(args.press_debug_log, cfg, every=int(args.press_debug_log_every))
        if args.press_debug_log
        else None
    )
    if press_debug_logger is not None:
        print(f"[INFO] press debug log enabled: {press_debug_logger.path}", flush=True)
        press_debug_logger.write_event(
            "after_reset",
            env,
            obs,
            extra={
                "argv": sys.argv,
                "robot_urdf": str(cfg.urdf_path),
                "robot_usd_output_dir": str(cfg.usd_output_dir or ""),
            },
        )
    if live_control is not None:
        live_control.apply(env)
    if presser_pose_recorder is not None:
        presser_pose_recorder.maybe_write(env, obs, -1, force=True)
    wait_for_press_setup_before_motion(env, simulation_app, presser_pose_recorder, press_debug_logger)
    sync_robot_hold = getattr(env, "sync_press_robot_hold_pose", None)
    if callable(sync_robot_hold):
        sync_robot_hold()
    sync_robot_root = getattr(env, "sync_press_robot_root_pose", None)
    if callable(sync_robot_root):
        sync_robot_root()
    if press_debug_logger is not None:
        press_debug_logger.reset_baseline()
        press_debug_logger.write_event("before_run_loop", env, obs)
    fots_adapter = None
    if args.enable_fots:
        fots_adapter = RevoFotsAdapter(
            RevoFotsCfg(
                width=240 // max(1, int(args.tacmap_resolution_step)),
                height=240 // max(1, int(args.tacmap_resolution_step)),
                contact_threshold_mm=float(args.fots_contact_threshold_mm),
                depth_scale=float(args.fots_depth_scale),
                lamb=(
                    float(args.fots_lamb_dilate),
                    float(args.fots_lamb_shear),
                    float(args.fots_lamb_twist),
                ),
                theta=float(args.fots_theta),
                track_contact_center=bool(args.fots_track_contact_center),
                show_depth_background=bool(args.fots_depth_background),
                marker_arrow_scale=float(args.fots_arrow_scale),
                marker_size=float(args.fots_marker_size),
                marker_size_depth_gain=float(args.fots_marker_size_depth_gain),
                depth_background_max_mm=(
                    float(args.fots_depth_background_max_mm)
                    if args.fots_depth_background_max_mm is not None
                    else float(args.tacmap_max_distance) * 1000.0
                ),
                device=str(getattr(env, "_device", "cuda")),
            )
        )
    tacex_rgb_adapter = None
    if args.enable_tacex_rgb:
        tacex_rgb_adapter = RevoTacExRgbAdapter(
            RevoTacExRgbCfg(
                width=int(args.tacex_rgb_width),
                height=int(args.tacex_rgb_height),
                depth_scale=float(args.tacex_rgb_depth_scale),
                with_shadow=bool(args.tacex_rgb_with_shadow),
                device=str(args.tacex_rgb_device or getattr(args, "device", "cuda")),
                calib_dir=str(args.tacex_rgb_calib_dir),
                taxim_sim_dir=str(args.tacex_rgb_sim_dir),
            )
        )
    tacsl_shear_adapter = None
    if args.enable_tacsl_shear:
        tacsl_shear_adapter = RevoTacslShearAdapter(
            RevoTacslShearCfg(
                normal_force_threshold=float(args.tacsl_shear_normal_force_threshold),
                shear_force_threshold=float(args.tacsl_shear_force_threshold),
                resolution=int(args.tacsl_shear_resolution),
                render_rows=int(args.tacsl_shear_render_rows),
                render_cols=int(args.tacsl_shear_render_cols),
                render_stride=int(args.tacsl_shear_render_stride),
                min_arrow_length_px=float(args.tacsl_shear_min_arrow_length_px),
                device=str(getattr(env, "_device", "cuda")),
            )
        )
    hydroshear_adapter = None
    if args.enable_hydroshear_marker:
        hydroshear_sample_mode = str(args.hydroshear_object_sample_mode)
        hydroshear_object_samples_l = sample_stage_object_surface_points(
            "/World/Plug",
            sample_count=int(args.hydroshear_object_sample_count),
            seed=int(args.hydroshear_object_sample_seed),
            sample_mode=hydroshear_sample_mode,
            poisson_radius=float(args.hydroshear_poisson_radius),
            poisson_initial_count=int(args.hydroshear_poisson_initial_count),
        )
        if hydroshear_object_samples_l is None:
            hydroshear_object_samples_l = sample_usd_mesh_surface_points(
                cfg.press_object_usd_path,
                sample_count=int(args.hydroshear_object_sample_count),
                seed=int(args.hydroshear_object_sample_seed),
                scale=float(getattr(cfg, "plug_scale", 1.0)),
                sample_mode=hydroshear_sample_mode,
                poisson_radius=float(args.hydroshear_poisson_radius),
                poisson_initial_count=int(args.hydroshear_poisson_initial_count),
            )
            if hydroshear_object_samples_l is not None:
                print("[INFO] HydroShear using fallback source-USD object samples", flush=True)
        hydroshear_adapter = RevoCurvedHydroShearAdapter(
            RevoCurvedHydroShearCfg(
                width=240 // max(1, int(args.tacmap_resolution_step)),
                height=240 // max(1, int(args.tacmap_resolution_step)),
                contact_threshold_mm=float(args.hydroshear_contact_threshold_mm),
                lambda_dilate=float(args.hydroshear_lamb_dilate),
                lambda_shear=float(args.hydroshear_lamb_shear),
                dilate_scale=float(args.hydroshear_dilate_scale),
                shear_scale=float(args.hydroshear_shear_scale),
                hydrosoft_k=float(args.hydroshear_hydrosoft_k),
                hydrosoft_e=float(args.hydroshear_hydrosoft_e),
                hydrosoft_area=float(args.hydroshear_hydrosoft_area),
                hydrosoft_mu=float(args.hydroshear_hydrosoft_mu),
                arrow_scale_px_per_m=float(args.hydroshear_arrow_scale),
                min_arrow_length_px=float(args.hydroshear_min_arrow_length_px),
                show_depth_background=bool(args.hydroshear_depth_background),
                depth_background_max_mm=float(args.hydroshear_depth_background_max_mm),
                object_sample_reference_count=4096,
                object_sample_points_l=hydroshear_object_samples_l,
                device=str(getattr(env, "_device", "cuda")),
            )
        )
    hydroshear_sample_viz = (
        HydroShearPointViz(
            path="/Visuals/HydroShearObjectSamples/env_0/ContactSamplesRed",
            color=(1.0, 0.0, 0.0),
            primary_attr="debug_all_object_sample_points_w",
            fallback_attr="debug_object_sample_points_w",
            radius=float(args.hydroshear_sample_point_radius),
            max_points=int(args.hydroshear_sample_point_max),
        )
        if bool(args.show_hydroshear_sample_points)
        else None
    )
    hydroshear_marker_viz = (
        HydroShearPointViz(
            path="/Visuals/HydroShearMarkers/env_0/MarkerPointsBlue",
            color=(0.0, 0.2, 1.0),
            primary_attr="debug_marker_points_w",
            radius=float(args.hydroshear_marker_point_radius),
            max_points=1000,
            sensor_index=display_sensor_index,
        )
        if bool(args.show_hydroshear_marker_points)
        else None
    )
    hydroshear_marker_normal_viz = (
        HydroShearVectorViz(
            path="/Visuals/HydroShearMarkers/env_0/MarkerNormalsYellow",
            color=(1.0, 1.0, 0.0),
            points_attr="debug_marker_points_w",
            vectors_attr="debug_marker_normals_w",
            length=float(args.hydroshear_marker_normal_length),
            width=float(args.hydroshear_marker_normal_width),
            max_vectors=1000,
            sensor_index=display_sensor_index,
        )
        if bool(args.show_hydroshear_marker_normals)
        else None
    )
    def make_hydroshear_marker_axis_viz(
        *,
        path: str,
        color: tuple[float, float, float],
        vectors_attr: str,
    ) -> HydroShearVectorViz:
        return HydroShearVectorViz(
            path=path,
            color=color,
            points_attr="debug_marker_points_w",
            vectors_attr=vectors_attr,
            length=float(args.hydroshear_marker_axis_length),
            width=float(args.hydroshear_marker_axis_width),
            max_vectors=int(args.hydroshear_marker_axis_max),
            sensor_index=display_sensor_index,
        )

    hydroshear_marker_col_axis_viz = (
        make_hydroshear_marker_axis_viz(
            path="/Visuals/HydroShearMarkers/env_0/MarkerColAxesCyan",
            color=(0.0, 1.0, 1.0),
            vectors_attr="debug_marker_col_axes_w",
        )
        if bool(args.show_hydroshear_marker_axes)
        else None
    )
    hydroshear_marker_row_axis_viz = (
        make_hydroshear_marker_axis_viz(
            path="/Visuals/HydroShearMarkers/env_0/MarkerRowAxesOrange",
            color=(1.0, 0.45, 0.0),
            vectors_attr="debug_marker_row_axes_w",
        )
        if bool(args.show_hydroshear_marker_axes)
        else None
    )
    tacmap_marker_overlay_fingers = tuple(
        finger for finger in active_fingers if finger in FINGER_MAPS
    )
    tacmap_finger_marker_point_viz = (
        TacMapFingerMarkerPointViz(
            fingers=tacmap_marker_overlay_fingers,
            root_path="/Visuals/TacMapFingerMarkers/env_0",
            color=(0.0, 0.2, 1.0),
            radius=float(args.hydroshear_marker_point_radius),
        )
        if bool(args.show_tacmap_finger_marker_points) and bool(tacmap_marker_overlay_fingers)
        else None
    )
    tacmap_finger_marker_normal_viz = (
        TacMapFingerMarkerNormalViz(
            fingers=tacmap_marker_overlay_fingers,
            root_path="/Visuals/TacMapFingerMarkers/env_0",
            color=(1.0, 1.0, 0.0),
            length=float(args.hydroshear_marker_normal_length),
            width=float(args.hydroshear_marker_normal_width),
        )
        if bool(args.show_tacmap_finger_marker_points)
        and bool(args.show_hydroshear_marker_normals)
        and bool(tacmap_marker_overlay_fingers)
        else None
    )
    if tacmap_finger_marker_point_viz is not None:
        print(
            f"[INFO] TacMap finger marker viz: fingers={tacmap_marker_overlay_fingers}, "
            f"source=calibrated Vitai pixels + camera extrinsics + rubber mesh",
            flush=True,
        )
    tacmap_finger_center_arrow_viz = (
        TacMapFingerCenterArrowViz(
            fingers=active_fingers,
            path="/Visuals/TacMapFingerCenters/env_0/CenterRayGreen",
            color=(0.0, 1.0, 0.0),
            length=float(args.tacmap_finger_center_arrow_length),
            width=float(args.tacmap_finger_center_arrow_width),
            head_length=float(args.tacmap_finger_center_arrow_head_length),
            head_width=float(args.tacmap_finger_center_arrow_head_width),
            center_radius=float(args.tacmap_finger_center_arrow_center_radius),
            use_mean_normal=bool(cfg.tacmap_link_surface_use_mean_normal),
            custom_ray_direction=cfg.tacmap_link_surface_ray_direction,
            press_start_offset=float(cfg.press_start_offset),
            press_end_offset=float(cfg.press_end_offset),
        )
        if legacy_integrated_run and not bool(args.hide_tacmap_finger_center_arrows)
        else None
    )
    tacmap_force_dt = float(getattr(cfg, "physics_dt", 1.0 / 120.0))
    tacmap_force_stiffness = (
        float(args.tacmap_normal_force_stiffness)
        if args.tacmap_normal_force_stiffness is not None
        else float(getattr(cfg, "compliant_contact_stiffness", 280.0))
    )
    tacmap_force_damping = (
        float(args.tacmap_normal_force_damping)
        if args.tacmap_normal_force_damping is not None
        else float(getattr(cfg, "compliant_contact_damping", 20.0))
    )
    print(
        f"[INFO] TacMap normal force estimate: Fn=max(0,k*depth+c*ddepth/dt), "
        f"k={tacmap_force_stiffness:g}, c={tacmap_force_damping:g}, dt={tacmap_force_dt:g}, "
        f"sample_area={float(args.tacmap_normal_force_sample_area):g}, "
        f"arrow_scale={float(args.tacmap_normal_force_arrow_scale):g}",
        flush=True,
    )
    tacmap_center_arrow_viz = (
        TacMapCenterArrowViz(
            path="/Visuals/TacMapContactCenter/env_0/CenterNormalRed",
            color=(1.0, 0.0, 0.0),
            length=float(args.tacmap_center_arrow_length),
            width=float(args.tacmap_center_arrow_width),
            head_length=float(args.tacmap_center_arrow_head_length),
            head_width=float(args.tacmap_center_arrow_head_width),
            center_radius=float(args.tacmap_center_arrow_center_radius),
            start_offset=float(args.tacmap_center_arrow_start_offset),
            sigma_px=float(args.tacmap_center_arrow_sigma_px),
            flip=bool(args.tacmap_center_arrow_flip),
            dt=tacmap_force_dt,
            stiffness=tacmap_force_stiffness,
            damping=tacmap_force_damping,
            force_arrow_scale=float(args.tacmap_normal_force_arrow_scale),
            min_arrow_length=float(args.tacmap_normal_force_min_arrow_length),
            sample_area=float(args.tacmap_normal_force_sample_area),
        )
        if not bool(args.hide_tacmap_center_arrow) and not bool(args.disable_tacmap)
        else None
    )

    show_local_ui = (not bool(getattr(args, "headless", True))) and not bool(args.no_local_ui)
    panel = None
    live_server = start_live_web_server(args.live_port) if args.live_web else None
    contact_benchmark_rows: list[dict[str, float | int | str]] = []
    pressure_trace = None
    pressure_verify_failed = False
    normal_ray_enabled = bool(args.enable_normal_ray_pressure or args.pressure_view_source == "normal_ray")
    geometry_normal_ray_enabled = bool(
        args.enable_geometry_normal_ray_pressure or args.pressure_view_source == "geometry_normal_ray"
    )
    pressure_trace_enabled = bool(args.save_pressure_trace or args.verify_pressure_trace)
    pressure_layout_arrays = (
        pressure_trace_layout_arrays(env, cfg)
        if (pressure_trace_enabled or normal_ray_enabled or geometry_normal_ray_enabled)
        else {}
    )
    normal_ray_alignment = None
    normal_ray_prev_penetration = None
    geometry_normal_ray_mesh_vertices = None
    geometry_normal_ray_mesh_triangles = None
    geometry_normal_ray_mesh_summary = None
    geometry_normal_ray_prev_penetration = None
    geometry_normal_ray_rest_baseline = None
    geometry_normal_ray_origin_source = "pressure_taxel_points_l_m"
    geometry_needs_surface_origin = bool(
        geometry_normal_ray_enabled
        and str(args.geometry_normal_ray_origin_source) == "tacmap_aligned"
        and "tacmap_grid_points_l_m" in pressure_layout_arrays
        and "tacmap_grid_axes_l" in pressure_layout_arrays
    )
    if normal_ray_enabled or geometry_needs_surface_origin:
        normal_ray_alignment = build_normal_ray_runtime_alignment(
            pressure_layout_arrays,
            distance_mode=str(args.normal_ray_pressure_distance_mode),
            max_distance_m=float(args.normal_ray_pressure_max_distance),
        )
        pressure_layout_arrays.update(
            {
                "normal_ray_alignment_source_index": np.asarray(normal_ray_alignment["source_index"]),
                "normal_ray_alignment_valid_mask": np.asarray(normal_ray_alignment["valid_mask"], dtype=np.uint8),
                "normal_ray_alignment_nn_distance_m": np.asarray(normal_ray_alignment["nn_distance_m"], dtype=np.float32),
            }
        )
    if geometry_needs_surface_origin and normal_ray_alignment is not None:
        origin_points, origin_valid = aligned_reference_points_for_pressure_taxels(
            pressure_layout_arrays["tacmap_grid_points_l_m"],
            normal_ray_alignment["source_index"],
            normal_ray_alignment["valid_mask"],
            fallback_points=pressure_layout_arrays["pressure_taxel_points_l_m"],
        )
        pressure_layout_arrays.update(
            {
                "geometry_normal_ray_origin_points_l_m": origin_points,
                "geometry_normal_ray_origin_valid_mask": np.asarray(origin_valid, dtype=np.uint8),
            }
        )
        geometry_normal_ray_origin_source = "tacmap_link_surface_aligned"
    if normal_ray_enabled:
        print(
            "[INFO] normal-ray pressure enabled: "
            f"distance_mode={args.normal_ray_pressure_distance_mode}, "
            f"max_distance={float(args.normal_ray_pressure_max_distance):g}m, "
            f"deadband={float(args.normal_ray_pressure_deadband):g}m, "
            f"valid_sensors={normal_ray_alignment['summary'].get('valid_sensor_indices')}",
            flush=True,
        )
    if geometry_normal_ray_enabled:
        geometry_usd_path = args.geometry_normal_ray_usd_path or cfg.press_object_usd_path
        if args.geometry_normal_ray_vertices_npy is not None or args.geometry_normal_ray_triangles_npy is not None:
            if args.geometry_normal_ray_vertices_npy is None or args.geometry_normal_ray_triangles_npy is None:
                raise ValueError(
                    "--geometry-normal-ray-vertices-npy and --geometry-normal-ray-triangles-npy "
                    "must be provided together"
                )
            geometry_normal_ray_mesh_vertices, geometry_normal_ray_mesh_triangles, geometry_normal_ray_mesh_summary = (
                load_npy_mesh_as_object_local_triangles(
                    args.geometry_normal_ray_vertices_npy,
                    args.geometry_normal_ray_triangles_npy,
                    scale=float(cfg.plug_scale),
                )
            )
            geometry_source_text = f"npy={args.geometry_normal_ray_vertices_npy},{args.geometry_normal_ray_triangles_npy}"
        else:
            geometry_normal_ray_mesh_vertices, geometry_normal_ray_mesh_triangles, geometry_normal_ray_mesh_summary = (
                load_usd_mesh_as_object_local_triangles(geometry_usd_path, scale=float(cfg.plug_scale))
            )
            geometry_source_text = f"usd={geometry_usd_path}"
        rest_text = (
            f"{float(args.geometry_normal_ray_rest_distance):g}m"
            if args.geometry_normal_ray_rest_distance is not None
            else "first_frame_hit_baseline"
        )
        print(
            "[INFO] geometry-normal-ray pressure enabled: "
            f"{geometry_source_text}, "
            f"vertices={geometry_normal_ray_mesh_summary['vertex_count']}, "
            f"triangles={geometry_normal_ray_mesh_summary['triangle_count']}, "
            f"watertight={geometry_normal_ray_mesh_summary.get('topology', {}).get('is_edge_watertight')}, "
            f"boundary_edges={geometry_normal_ray_mesh_summary.get('topology', {}).get('boundary_edge_count')}, "
            f"nonmanifold_edges={geometry_normal_ray_mesh_summary.get('topology', {}).get('nonmanifold_edge_count')}, "
            f"mode={args.geometry_normal_ray_mode}, "
            f"max_distance={float(args.geometry_normal_ray_max_distance):g}m, "
            f"deadband={float(args.geometry_normal_ray_deadband):g}m, "
            f"samples={args.geometry_normal_ray_taxel_samples}, "
            f"min_support={float(args.geometry_normal_ray_sample_min_support_fraction):g}, "
            f"rest={rest_text}, "
            f"origin={geometry_normal_ray_origin_source}",
            flush=True,
        )
    if pressure_trace_enabled:
        pressure_layout_source = "urdf_pressure_layout" if args.pressure_layout_urdf is not None else "legacy_finger_map"
        row_distance_m, col_distance_m = cfg_pressure_grid_distances(cfg)
        pressure_trace = PressureTraceRecorder(
            args.pressure_trace_dir,
            {
                "pressure_trace_schema_version": "pressure_trace_v1",
                "pressure_backend_id": f"warpsdf_{pressure_contact_model}",
                "pressure_contact_model": pressure_contact_model,
                "pressure_surface_gap_mode": str(cfg.mesh_unsigned_contact_mode),
                "pressure_mesh_shell_thickness_m": float(cfg.mesh_shell_thickness),
                "pressure_layout_id": (
                    f"{pressure_layout_source}:{cfg.press_touch_link}:{cfg.num_rows}x{cfg.num_cols}:"
                    f"{row_distance_m:.9g}x{col_distance_m:.9g}m"
                ),
                "pressure_calibration_id": str(args.pressure_calib or "inline_cli_defaults"),
                "pressure_layout_source": pressure_layout_source,
                "pressure_layout_urdf": str(args.pressure_layout_urdf or ""),
                "pressure_layout_link": str(args.pressure_layout_link or ""),
                "pressure_layout_name": (
                    selected_pressure_pad_spec.map_name if selected_pressure_pad_spec is not None else ""
                ),
                "pressure_origin_semantics": (
                    selected_pressure_pad_spec.origin_semantics
                    if selected_pressure_pad_spec is not None
                    else "pad_surface"
                ),
                "pressure_layout_declared_taxel_count": (
                    int(selected_pressure_pad_spec.taxel_count)
                    if selected_pressure_pad_spec is not None and selected_pressure_pad_spec.taxel_count is not None
                    else None
                ),
                "pressure_taxel_count": int(cfg.num_rows) * int(cfg.num_cols),
                "robot_urdf": str(cfg.urdf_path),
                "robot_usd_output_dir": str(cfg.usd_output_dir or ""),
                "press_touch_link": str(cfg.press_touch_link),
                "press_motion_source": press_motion_source_metadata,
                "press_object_control": str(cfg.press_object_control),
                "press_motion_actor": str(cfg.press_motion_actor),
                "press_hand_axis_link": str(cfg.press_hand_axis_link),
                "press_hand_axis_l": [float(value) for value in cfg.press_hand_axis_l],
                "presser_collision_enabled": bool(cfg.press_object_collision_enabled),
                "press_distance_m": float(cfg.press_start_offset - cfg.press_end_offset),
                "press_distance_cli_m": float(args.press_distance) if args.press_distance is not None else None,
                "press_finger_joints": [str(value) for value in cfg.press_finger_joints],
                "press_finger_start_rad": float(cfg.press_finger_start_rad),
                "press_finger_end_rad": float(cfg.press_finger_end_rad),
                "press_finger_kinematic_sensor_pose": bool(cfg.press_finger_kinematic_sensor_pose),
                "press_hold_joint_pose": str(cfg.press_hold_joint_pose),
                "press_setup_before_motion": bool(args.press_setup_before_motion),
                "press_setup_start_file": str(args.press_setup_start_file or ""),
                "presser_initial_pose_world_xyz_xyzw": [float(value) for value in cfg.plug_default_pose],
                "presser_pose_save_file": str(args.presser_pose_save_file or ""),
                "press_center_l": [float(value) for value in cfg.press_center_l] if cfg.press_center_l is not None else None,
                "press_normal_l": [float(value) for value in cfg.press_normal_l] if cfg.press_normal_l is not None else None,
                "press_center_cli_l": (
                    [float(value) for value in args.press_center_l] if args.press_center_l is not None else None
                ),
                "press_center_offset_cli_l": (
                    [float(value) for value in args.press_center_offset_l]
                    if args.press_center_offset_l is not None
                    else None
                ),
                "press_normal_cli_l": (
                    [float(value) for value in args.press_normal_l] if args.press_normal_l is not None else None
                ),
                "live_control_file": str(args.live_control_file or ""),
                "press_patch_pos_l": (
                    [float(value) for value in cfg.press_patch_pos_l]
                    if getattr(cfg, "press_patch_pos_l", None) is not None
                    else None
                ),
                "press_patch_quat_l": (
                    [float(value) for value in cfg.press_patch_quat_l]
                    if getattr(cfg, "press_patch_quat_l", None) is not None
                    else None
                ),
                "press_tacmap_files_enabled": bool(getattr(cfg, "press_tacmap_files_enabled", True)),
                "touch_collision_paths": [str(value) for value in getattr(cfg, "touch_collision_paths", ())],
                "mode": args.mode,
                "legacy_finger_arg": args.finger,
                "fingers_arg": str(args.fingers or ""),
                "focus_finger_arg": str(args.focus_finger or ""),
                "focus_finger": str(primary_finger),
                "primary_finger": str(primary_finger),
                "active_fingers": [str(value) for value in active_fingers],
                "active_pressure_labels": list(labels),
                "presser": args.presser,
                "tacmap_ray_mode": cfg.tacmap_ray_mode,
                "pressure_calib": str(args.pressure_calib or ""),
                "pressure_calib_overrides": {
                    key: calibration_value_metadata(value) for key, value in pressure_calib_overrides.items()
                },
                "pressure_stiffness": calibration_value_metadata(cfg.stiffness),
                "pressure_damping": calibration_value_metadata(cfg.damping),
                "pressure_max_force": calibration_value_metadata(cfg.max_force),
                "pressure_gain": calibration_value_metadata(cfg.pressure_gain),
                "pressure_bias": calibration_value_metadata(cfg.pressure_bias),
                "pressure_gamma": calibration_value_metadata(cfg.pressure_gamma),
                "pressure_threshold": calibration_value_metadata(cfg.pressure_threshold),
                "taxel_area": calibration_value_metadata(cfg.taxel_area),
                "num_rows": int(cfg.num_rows),
                "num_cols": int(cfg.num_cols),
                "point_distance_m": float(cfg.point_distance),
                "row_distance_m": float(row_distance_m),
                "col_distance_m": float(col_distance_m),
                "pressure_layout_pad_size_m": (
                    list(selected_pressure_pad_spec.pad_size)
                    if selected_pressure_pad_spec is not None and selected_pressure_pad_spec.pad_size is not None
                    else None
                ),
                "pressure_layout_pad_size_semantics": (
                    selected_pressure_pad_spec.pad_size_semantics
                    if selected_pressure_pad_spec is not None
                    else ""
                ),
                "pressure_layout_origin_xyz_m": (
                    list(selected_pressure_pad_spec.origin_xyz)
                    if selected_pressure_pad_spec is not None
                    else None
                ),
                "pressure_layout_origin_rpy_rad": (
                    list(selected_pressure_pad_spec.origin_rpy)
                    if selected_pressure_pad_spec is not None
                    else None
                ),
                "normal_axis": int(cfg.normal_axis),
                "normal_offset_m": float(cfg.normal_offset),
                "normal_sign": (
                    float(cfg.normal_sign)
                    if getattr(cfg, "normal_sign", None) is not None
                    else None
                ),
                "penetration_deadband_m": float(cfg.penetration_deadband),
                "physx_contact_map": bool(cfg.enable_physx_contact_force_map),
                "warpsdf_contact_gate": bool(cfg.gate_warpsdf_with_physx_contact),
                "physx_contact_usage": "sparse_sanity_debug_only",
                "warpsdf_contact_gate_usage": "experimental_debug_not_l1_acceptance",
                "pressure_view_source": str(args.pressure_view_source),
                "normal_ray_pressure_enabled": bool(normal_ray_enabled),
                "normal_ray_pressure_deadband_m": float(args.normal_ray_pressure_deadband),
                "normal_ray_pressure_alignment": (
                    normal_ray_alignment["summary"] if normal_ray_alignment is not None else None
                ),
                "geometry_normal_ray_pressure_enabled": bool(geometry_normal_ray_enabled),
                "geometry_normal_ray_mode": str(args.geometry_normal_ray_mode),
                "geometry_normal_ray_requested_origin_source": str(args.geometry_normal_ray_origin_source),
                "geometry_normal_ray_origin_source": str(geometry_normal_ray_origin_source),
                "geometry_normal_ray_deadband_m": float(args.geometry_normal_ray_deadband),
                "geometry_normal_ray_max_distance_m": float(args.geometry_normal_ray_max_distance),
                "geometry_normal_ray_taxel_samples": str(args.geometry_normal_ray_taxel_samples),
                "geometry_normal_ray_sample_spacing_m": (
                    None
                    if args.geometry_normal_ray_sample_spacing is None
                    else float(args.geometry_normal_ray_sample_spacing)
                ),
                "geometry_normal_ray_sample_aggregation": str(args.geometry_normal_ray_sample_aggregation),
                "geometry_normal_ray_sample_min_support_fraction": float(
                    args.geometry_normal_ray_sample_min_support_fraction
                ),
                "geometry_normal_ray_rest_distance_m": (
                    None
                    if args.geometry_normal_ray_rest_distance is None
                    else float(args.geometry_normal_ray_rest_distance)
                ),
                "geometry_normal_ray_rest_mode": (
                    "first_frame_hit_baseline"
                    if args.geometry_normal_ray_rest_distance is None
                    else "constant"
                ),
                "geometry_normal_ray_mesh": geometry_normal_ray_mesh_summary,
                "layout_arrays": {key: list(value.shape) for key, value in pressure_layout_arrays.items()},
                "tacmap_link_surface": (
                    {
                        "height": int(cfg.tacmap_link_surface_height),
                        "width": int(cfg.tacmap_link_surface_width),
                        "ray_axis": str(cfg.tacmap_link_surface_ray_axis),
                        "ray_direction": (
                            [float(v) for v in cfg.tacmap_link_surface_ray_direction]
                            if cfg.tacmap_link_surface_ray_direction is not None
                            else None
                        ),
                        "grid_u_axis": str(cfg.tacmap_link_surface_grid_u_axis),
                        "grid_v_axis": str(cfg.tacmap_link_surface_grid_v_axis),
                        "grid_u_size_m": float(cfg.tacmap_link_surface_grid_u_size),
                        "grid_v_size_m": float(cfg.tacmap_link_surface_grid_v_size),
                        "grid_center_l_m": [float(v) for v in cfg.tacmap_link_surface_grid_center],
                        "max_distance_m": float(cfg.tacmap_max_distance),
                    }
                    if str(cfg.tacmap_ray_mode) == "link_surface"
                    else None
                ),
            },
            layout_arrays=pressure_layout_arrays,
        )
        print(f"[INFO] pressure trace enabled: {pressure_trace.out_dir / (pressure_trace.run_id + '.npz')}", flush=True)

    total = int(cfg.press_steps) if args.mode == "press" else 10**12
    if args.max_steps > 0:
        total = int(args.max_steps)
    elif args.mode in ("points", "press"):
        total = 10**12

    for step in range(total):
        if not simulation_app.is_running():
            break
        if live_control is not None:
            live_control.apply(env)
        action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, *_ = env.step(action)
        if presser_pose_recorder is not None:
            presser_pose_recorder.maybe_write(env, obs, step)
        if press_debug_logger is not None:
            press_debug_logger.maybe_write(phase="run", step=step, env=env, obs=obs)

        force = obs.get("pressure_force_map", obs["tactile"])
        pressure_force_raw = obs.get("pressure_force_map_raw")
        pressure_penetration = obs.get("pressure_penetration_map")
        pressure_signed_distance = obs.get("pressure_signed_distance_map")
        pressure_penetration_velocity = obs.get("pressure_penetration_velocity_map")
        physx_contact_force = obs.get("physx_contact_force_map")
        physx_contact_force_raw = obs.get("physx_contact_force_map_raw")
        physx_contact_count = obs.get("physx_contact_count")
        tacmap = obs["tacmap"]
        tacmap_raw = obs.get("tacmap_raw")
        tacmap_surface_raw = obs.get("tacmap_surface_raw")
        tacmap_object_raw = obs.get("tacmap_object_raw")
        tacmap_surface_points_w = obs.get("tacmap_surface_points_w")
        tacmap_surface_normals_w = obs.get("tacmap_surface_normals_w")
        tacmap_surface_valid = obs.get("tacmap_surface_valid")
        tacmap_object_points_w = obs.get("tacmap_object_points_w")
        tacmap_object_valid = obs.get("tacmap_object_valid")
        tacsl_penetration_depth = obs.get("tacsl_penetration_depth")
        tacsl_normal_force = obs.get("tacsl_normal_force")
        tacsl_shear_force = obs.get("tacsl_shear_force")
        fots_theta = obs.get("fots_theta")
        plug_pose = obs.get("plug_pose")
        press_touch_pose = obs.get("press_touch_pose")
        force_tensor = pressure_force_map_tensor_from_env(env, force)
        tacmap_raw_tensor = tacmap_raw_tensor_from_env(env) if cfg.enable_tacmap else None
        tacmap_surface_raw_tensor = tacmap_surface_raw_tensor_from_env(env) if cfg.enable_tacmap else None
        tacmap_geometry_tensors = tacmap_link_surface_geometry_tensors_from_env(env) if cfg.enable_tacmap else None
        if tacmap_geometry_tensors is not None:
            (
                tacmap_surface_points_w_tensor,
                tacmap_surface_normals_w_tensor,
                tacmap_surface_valid_tensor,
                tacmap_object_points_w_tensor,
                tacmap_object_valid_tensor,
            ) = tacmap_geometry_tensors
        else:
            tacmap_surface_points_w_tensor = tacmap_surface_points_w
            tacmap_surface_normals_w_tensor = tacmap_surface_normals_w
            tacmap_surface_valid_tensor = tacmap_surface_valid
            tacmap_object_points_w_tensor = tacmap_object_points_w
            tacmap_object_valid_tensor = tacmap_object_valid
        force_for_display = force_tensor if force_tensor is not None else force
        tacmap_raw_for_compute = tacmap_raw_tensor if tacmap_raw_tensor is not None else tacmap_raw
        tacmap_surface_raw_for_compute = (
            tacmap_surface_raw_tensor if tacmap_surface_raw_tensor is not None else tacmap_surface_raw
        )
        fots_output = (
            fots_adapter.step(tacmap_raw_for_compute, theta_rad=fots_theta)
            if fots_adapter is not None and tacmap_raw_for_compute is not None
            else None
        )
        tacex_rgb_output = (
            tacex_rgb_adapter.step(tacmap_raw_for_compute)
            if tacex_rgb_adapter is not None and tacmap_raw_for_compute is not None
            else None
        )
        normal_ray_output, normal_ray_prev_penetration = (
            normal_ray_pressure_from_tacmap(
                tacmap_raw,
                normal_ray_alignment,
                cfg=cfg,
                previous_penetration=normal_ray_prev_penetration,
                contact_deadband_m=float(args.normal_ray_pressure_deadband),
            )
            if normal_ray_enabled
            else (None, normal_ray_prev_penetration)
        )
        normal_ray_force = normal_ray_output["pressure_norm"] if normal_ray_output is not None else None
        geometry_normal_ray_output, geometry_normal_ray_prev_penetration, geometry_normal_ray_rest_baseline = (
            geometry_normal_ray_pressure_from_mesh(
                pressure_layout_arrays,
                object_vertices_l=geometry_normal_ray_mesh_vertices,
                triangles=geometry_normal_ray_mesh_triangles,
                object_pose_w=plug_pose,
                touch_pose_w=press_touch_pose,
                cfg=cfg,
                previous_penetration=geometry_normal_ray_prev_penetration,
                rest_distance_m=args.geometry_normal_ray_rest_distance,
                rest_baseline_m=geometry_normal_ray_rest_baseline,
                contact_deadband_m=float(args.geometry_normal_ray_deadband),
                max_distance_m=float(args.geometry_normal_ray_max_distance),
                mode=str(args.geometry_normal_ray_mode),
                taxel_sample_pattern=str(args.geometry_normal_ray_taxel_samples),
                taxel_sample_spacing_m=args.geometry_normal_ray_sample_spacing,
                sample_aggregation=str(args.geometry_normal_ray_sample_aggregation),
                sample_min_support_fraction=float(args.geometry_normal_ray_sample_min_support_fraction),
            )
            if geometry_normal_ray_enabled
            else (None, geometry_normal_ray_prev_penetration, geometry_normal_ray_rest_baseline)
        )
        geometry_normal_ray_force = (
            geometry_normal_ray_output["pressure_norm"] if geometry_normal_ray_output is not None else None
        )
        if args.pressure_view_source == "normal_ray" and normal_ray_force is not None:
            display_force = normal_ray_force
        elif args.pressure_view_source == "geometry_normal_ray" and geometry_normal_ray_force is not None:
            display_force = geometry_normal_ray_force
        else:
            display_force = force_for_display
        display_force_pane = sensor_display_batch(display_force, display_sensor_index)
        f_img = force_strip(display_force_pane, scale=int(args.force_scale), gamma=float(args.force_gamma))
        focus_pressure_pad_panes = []
        focus_pressure_pad_tile_size = int(args.focus_pressure_pad_tile_size)
        for sensor_index in focus_pressure_pad_sensor_indices:
            pane = force_strip(
                sensor_display_batch(force_for_display, sensor_index),
                scale=int(args.force_scale),
                gamma=float(args.force_gamma),
            )
            if focus_pressure_pad_tile_size > 0:
                pane = _resize_nearest_to_shape(pane, focus_pressure_pad_tile_size, focus_pressure_pad_tile_size)
            focus_pressure_pad_panes.append(pane)
        if not pressure_pad_run:
            f_img = force_strip(np.zeros_like(display_force_pane) if not isinstance(display_force_pane, torch.Tensor) else torch.zeros_like(display_force_pane), scale=int(args.force_scale), gamma=float(args.force_gamma))
            if focus_pressure_pad_panes:
                f_img = None
        physx_contact_img = (
            force_strip(
                sensor_display_batch(physx_contact_force, display_sensor_index),
                scale=int(args.force_scale),
                gamma=float(args.force_gamma),
            )
            if cfg.enable_physx_contact_force_map and physx_contact_force is not None
            else None
        )
        tacsl_shear_output = (
            tacsl_shear_adapter.step(tacsl_normal_force, tacsl_shear_force, tacsl_penetration_depth)
            if tacsl_shear_adapter is not None
            and tacsl_normal_force is not None
            and tacsl_shear_force is not None
            else None
        )
        hydroshear_output = (
            hydroshear_adapter.step(
                tacmap_raw_for_compute,
                tacmap_surface_points_w_tensor,
                tacmap_surface_valid_tensor,
                tacmap_object_points_w_tensor,
                tacmap_object_valid_tensor,
                plug_pose,
                tacmap_surface_raw_for_compute,
                surface_normals_w=tacmap_surface_normals_w_tensor,
            )
            if hydroshear_adapter is not None
            and tacmap_raw_for_compute is not None
            and tacmap_surface_points_w_tensor is not None
            and tacmap_surface_valid_tensor is not None
            and tacmap_object_points_w_tensor is not None
            and tacmap_object_valid_tensor is not None
            else None
        )
        if hydroshear_output is not None and hasattr(env, "update_rl_hydroshear_obs"):
            env.update_rl_hydroshear_obs(obs, hydroshear_output.displacement_m)
        if tacmap_finger_marker_point_viz is not None:
            tacmap_finger_marker_point_viz.update(env)
        if tacmap_finger_marker_normal_viz is not None:
            tacmap_finger_marker_normal_viz.update(env)
        if tacmap_finger_center_arrow_viz is not None:
            tacmap_finger_center_arrow_viz.update(env)
        if hydroshear_sample_viz is not None:
            hydroshear_sample_viz.update(hydroshear_output)
        if hydroshear_marker_viz is not None:
            hydroshear_marker_viz.update(hydroshear_output)
        if hydroshear_marker_normal_viz is not None:
            hydroshear_marker_normal_viz.update(hydroshear_output)
        if hydroshear_marker_col_axis_viz is not None:
            hydroshear_marker_col_axis_viz.update(hydroshear_output)
        if hydroshear_marker_row_axis_viz is not None:
            hydroshear_marker_row_axis_viz.update(hydroshear_output)
        if tacmap_center_arrow_viz is not None:
            tacmap_center_arrow_viz.update(
                tacmap_raw,
                tacmap_surface_points_w,
                tacmap_surface_normals_w,
                tacmap_surface_valid,
            )
        t_img = None
        if cfg.enable_tacmap:
            tacmap_display_max_m = (
                float(args.tacmap_display_max_mm) * 1.0e-3
                if args.tacmap_display_max_mm is not None
                else float(args.tacmap_max_distance)
            )
            tacmap_display = sensor_display_batch(tacmap, display_sensor_index)
            tacmap_raw_display = sensor_display_batch(tacmap_raw_for_compute, display_sensor_index)
            t_img = tacmap_strip(
                tacmap_display,
                tacmap_raw=tacmap_raw_display,
                scale=int(args.tacmap_scale),
                gamma=float(args.tacmap_gamma),
                view=str(args.tacmap_view),
                display_max_m=tacmap_display_max_m,
            )
        fots_panes = []
        if fots_output is not None:
            if args.fots_view in ("flow", "both"):
                fots_panes.append(image_strip(sensor_display_batch(fots_output.marker_images, display_sensor_index)))
            if args.fots_view in ("markers", "both"):
                fots_panes.append(image_strip(sensor_display_batch(fots_output.marker_overlay_images, display_sensor_index)))
        tacex_rgb_pane = (
            image_strip(sensor_display_batch(tacex_rgb_output.tactile_rgb, display_sensor_index))
            if tacex_rgb_output is not None
            else None
        )
        tacsl_shear_pane = (
            image_strip(sensor_display_batch(tacsl_shear_output.shear_images, display_sensor_index))
            if tacsl_shear_output is not None
            else None
        )
        hydroshear_original_pane = (
            image_strip(sensor_display_batch(hydroshear_output.original_marker_images, display_sensor_index))
            if hydroshear_output is not None
            else None
        )
        hydroshear_pane = (
            image_strip(sensor_display_batch(hydroshear_output.marker_images, display_sensor_index))
            if hydroshear_output is not None
            else None
        )
        hydroshear_debug_pane = hydroshear_debug_image(hydroshear_output, sensor_index=display_sensor_index)
        if args.hydroshear_debug_only:
            img = hydroshear_debug_pane if hydroshear_debug_pane is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            img = combined_image(
                *focus_pressure_pad_panes,
                f_img,
                physx_contact_img,
                t_img,
                *fots_panes,
                tacex_rgb_pane,
                tacsl_shear_pane,
                hydroshear_original_pane,
                hydroshear_pane,
            )
        stats_parts = []
        if pressure_pad_run:
            stats_parts.append(stats_line(display_force, tacmap if cfg.enable_tacmap else None, labels))
        normal_ray_stats = normal_ray_stats_line(normal_ray_force, labels)
        if normal_ray_stats:
            stats_parts.append(normal_ray_stats)
        geometry_normal_ray_stats = geometry_normal_ray_stats_line(geometry_normal_ray_force, labels)
        if geometry_normal_ray_stats:
            stats_parts.append(geometry_normal_ray_stats)
        if cfg.enable_physx_contact_force_map:
            physx_contact_stats = physx_contact_stats_line(
                physx_contact_force,
                physx_contact_force_raw,
                physx_contact_count,
                labels,
            )
            if physx_contact_stats:
                stats_parts.append(physx_contact_stats)
        raw_stats = tacmap_raw_stats_line(tacmap_raw if cfg.enable_tacmap else None, labels)
        if raw_stats:
            stats_parts.append(raw_stats)
        link_surface_stats = link_surface_raw_stats_line(
            tacmap_surface_raw if cfg.enable_tacmap else None,
            tacmap_object_raw if cfg.enable_tacmap else None,
            labels,
        )
        if link_surface_stats:
            stats_parts.append(link_surface_stats)
        press_depth_stats = press_depth_stats_line(
            press_depth_summary_from_env(env, pressure_penetration, plug_pose=plug_pose)
        )
        if press_depth_stats:
            stats_parts.append(press_depth_stats)
        fots_stats = fots_stats_line(fots_output, labels)
        if fots_stats:
            stats_parts.append(fots_stats)
        tacex_rgb_stats = tacex_rgb_stats_line(tacex_rgb_output, labels)
        if tacex_rgb_stats:
            stats_parts.append(tacex_rgb_stats)
        tacsl_shear_stats = tacsl_shear_stats_line(tacsl_shear_output, labels)
        if tacsl_shear_stats:
            stats_parts.append(tacsl_shear_stats)
        hydroshear_stats = hydroshear_marker_stats_line(hydroshear_output, labels)
        if hydroshear_stats:
            stats_parts.append(hydroshear_stats)
        if tacmap_center_arrow_viz is not None:
            stats_parts.append(tacmap_center_arrow_viz.stats_line(labels))
        stats = f"[step {step:06d}] " + ", ".join(part for part in stats_parts if part)

        if args.contact_benchmark_out:
            contact_benchmark_rows.append(
                contact_benchmark_row(
                    step=step,
                    cfg=cfg,
                    force=force,
                    pressure_force_raw=pressure_force_raw,
                    physx_contact_force=physx_contact_force,
                    physx_contact_force_raw=physx_contact_force_raw,
                    physx_contact_count=physx_contact_count,
                    tacmap=tacmap,
                    tacmap_raw=tacmap_raw,
                    sensor_index=int(args.contact_benchmark_sensor),
                    force_threshold=float(args.contact_benchmark_force_threshold),
                    depth_threshold_m=float(args.contact_benchmark_depth_threshold_mm) * 1.0e-3,
                )
            )

        if pressure_trace is not None:
            pressure_trace.record(
                step=step,
                pressure_penetration_m=pressure_penetration,
                pressure_signed_distance_m=pressure_signed_distance,
                pressure_penetration_velocity_mps=pressure_penetration_velocity,
                pressure_raw_n=pressure_force_raw,
                pressure_norm=force,
                tacmap_raw_m=tacmap_raw,
                physx_contact_count=physx_contact_count,
                normal_ray_penetration_m=(
                    normal_ray_output["penetration_m"] if normal_ray_output is not None else None
                ),
                normal_ray_signed_distance_m=(
                    normal_ray_output["signed_distance_m"] if normal_ray_output is not None else None
                ),
                normal_ray_penetration_velocity_mps=(
                    normal_ray_output["penetration_velocity_mps"] if normal_ray_output is not None else None
                ),
                normal_ray_pressure_raw_n=(
                    normal_ray_output["pressure_raw_n"] if normal_ray_output is not None else None
                ),
                normal_ray_pressure_norm=(
                    normal_ray_output["pressure_norm"] if normal_ray_output is not None else None
                ),
                normal_ray_total_force_n=(
                    normal_ray_output["total_force_n"] if normal_ray_output is not None else None
                ),
                normal_ray_center_of_pressure_px=(
                    normal_ray_output["center_of_pressure_px"] if normal_ray_output is not None else None
                ),
                geometry_normal_ray_penetration_m=(
                    geometry_normal_ray_output["penetration_m"] if geometry_normal_ray_output is not None else None
                ),
                geometry_normal_ray_signed_distance_m=(
                    geometry_normal_ray_output["signed_distance_m"] if geometry_normal_ray_output is not None else None
                ),
                geometry_normal_ray_penetration_velocity_mps=(
                    geometry_normal_ray_output["penetration_velocity_mps"]
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_pressure_raw_n=(
                    geometry_normal_ray_output["pressure_raw_n"] if geometry_normal_ray_output is not None else None
                ),
                geometry_normal_ray_pressure_norm=(
                    geometry_normal_ray_output["pressure_norm"] if geometry_normal_ray_output is not None else None
                ),
                geometry_normal_ray_total_force_n=(
                    geometry_normal_ray_output["total_force_n"] if geometry_normal_ray_output is not None else None
                ),
                geometry_normal_ray_center_of_pressure_px=(
                    geometry_normal_ray_output["center_of_pressure_px"]
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_support_fraction=(
                    geometry_normal_ray_output.get("sample_support_fraction")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_active_count=(
                    geometry_normal_ray_output.get("sample_active_count")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_mean_penetration_m=(
                    geometry_normal_ray_output.get("sample_mean_penetration_m")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_positive_mean_penetration_m=(
                    geometry_normal_ray_output.get("sample_positive_mean_penetration_m")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_max_penetration_m=(
                    geometry_normal_ray_output.get("sample_max_penetration_m")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_penetrations_m=(
                    geometry_normal_ray_output.get("sample_penetrations_m")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_offsets_l=(
                    geometry_normal_ray_output.get("sample_offsets_l")
                    if geometry_normal_ray_output is not None
                    else None
                ),
                geometry_normal_ray_sample_points_l_m=(
                    geometry_normal_ray_output.get("sample_points_l_m")
                    if geometry_normal_ray_output is not None
                    else None
                ),
            )

        if show_local_ui:
            if panel is None:
                panel = _IntegratedPanel(img.shape[0], img.shape[1])
            panel.update(img, stats)
        update_live_web(img, stats)
        maybe_show_cv(img)
        maybe_save(img, step)
        if tacmap_raw is not None:
            maybe_save_heightmap(tacmap_raw, step)
        maybe_save_fots(fots_output, step)
        maybe_save_tacex_rgb(tacex_rgb_output, step)
        maybe_save_tacsl_shear(tacsl_shear_output, step)
        maybe_save_hydroshear_marker(hydroshear_output, step)

        should_print = step % max(1, int(args.print_every)) == 0 or step == total - 1
        if should_print:
            print(stats, flush=True)

    write_contact_benchmark(contact_benchmark_rows, args.contact_benchmark_out)
    if presser_pose_recorder is not None:
        presser_pose_recorder.maybe_write(env, obs, total, force=True)
    if press_debug_logger is not None:
        press_debug_logger.write_event("finished", env, obs, extra={"total_steps_requested": int(total)})
        press_debug_logger.close()
    if pressure_trace is not None:
        trace_paths = pressure_trace.close()
        if trace_paths is not None:
            npz_path, metadata_path = trace_paths
            print(f"[TRACE] pressure trace: {npz_path}", flush=True)
            print(f"[TRACE] pressure metadata: {metadata_path}", flush=True)
            if args.verify_pressure_trace:
                pressure_verify_failed = not write_pressure_trace_verification(npz_path)
    env.close()
    if live_server is not None:
        live_server.shutdown()
    simulation_app.close()
    if pressure_verify_failed and args.pressure_verify_fail_on_threshold:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
