# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Revo3 RL tactile observations with a presser.

This entrypoint creates the normal ManagerBased RL environment, holds or tracks
the robot with a selectable mode, places the RL ``/Object`` presser near either
the focus finger's TacMap surface or one pressure pad, and displays the RL
observation tensors. ``--press-motion-actor hand`` uses Sharpa-style joint
targets instead of moving the robot root.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATE_ROOT = REPO_ROOT / "integrate"
TACMAP_ROOT = REPO_ROOT / "tacmap"

for path in (REPO_ROOT, REPO_ROOT / "source" / "BrainCo_DexHand", INTEGRATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tensor_image_utils import (  # noqa: E402
    jet_colormap,
    normalize_grid,
    resize_nearest_to_shape,
    tacmap_strip,
    to_numpy_uint8_rgb,
)
from tacsl_shear_adapter import RevoTacslShearAdapter, RevoTacslShearCfg  # noqa: E402
from fots_adapter import RevoFotsAdapter, RevoFotsCfg  # noqa: E402


FINGER_CHOICES = ("middle", "index", "ring", "pinky", "thumb")
PRESSURE_PAD_SEGMENTS = ("mcp", "pip", "palm")
PRESSURE_PAD_FINGER_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
FINGER_JOINT_NAME_NEEDLES = {
    "middle": ("right_mid",),
    "index": ("right_index",),
    "ring": ("right_ring",),
    "pinky": ("right_pinky",),
    "thumb": ("right_thumb",),
}
FINGER_ROLL_JOINT_STEMS = {
    "middle": "mid",
    "index": "index",
    "ring": "ring",
    "pinky": "pinky",
    "thumb": "thumb",
}
PRESSER_SPECS = {
    "task_object": {
        "usd": None,
        "default_flip_axis": "none",
        "default_extra_rot_axis": "none",
        "default_extra_rot_deg": 0.0,
        "rot": (1.0, 0.0, 0.0, 0.0),
    },
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
        "usd": TACMAP_ROOT / "assets" / "presser" / "ball_probe.usd",
        "default_flip_axis": "none",
        "default_extra_rot_axis": "x",
        "default_extra_rot_deg": 90.0,
        "rot": (1.0, 0.0, 0.0, 0.0),
    },
}
SQUARE_4_MESH_BBOX_L = (
    (-0.020999999716877937, -0.0209282748401165, 0.0),
    (0.020999999716877937, 0.0209282748401165, 0.01899999938905239),
)
BIMANUAL_URDF = REPO_ROOT / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Bimanual.urdf"
BIMANUAL_USD_DIR = Path("/tmp/bc_tactile_lab_bimanual_usd")
BIMANUAL_ROOT_POS = (0.4, 0.0, 0.495)
BIMANUAL_ROOT_ROT = (0.5, -0.5, 0.5, -0.5)
VITAI_MARKER_LAYOUT = (
    REPO_ROOT
    / "assets"
    / "revo21_right_touch"
    / "marker_positions"
    / "vitai_4fingers"
    / "marker_positions.npz"
)
VITAI_RIGHT_URDF = (
    REPO_ROOT
    / "assets"
    / "revo21_right_touch"
    / "urdf"
    / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
)
VITAI_CAMERA_INTRINSICS = REPO_ROOT / "vitai_4Fingers-320*240" / "camera_intrinsics_320x240.yaml"
VITAI_MARKER_BACKGROUND = (
    REPO_ROOT
    / "vitai_4Fingers-320*240"
    / "marker_annotations"
    / "reference_median_markerless.png"
)
VITAI_MARKER_RGB_BACKGROUND = VITAI_MARKER_BACKGROUND.with_name("reference_median.png")
VISUAL_LOCAL_TACMAP_OBJECT_SENSOR = "visual_local_tacmap_object_s"
VISUAL_LOCAL_TACMAP_SURFACE_SENSOR = "visual_local_tacmap_surface_ref_s"
VISUAL_LOCAL_TACMAP_RAY_COUNT = 1000
VISUAL_LOCAL_TACMAP_GAUSSIAN_KERNEL_SIZE = 9
VISUAL_LOCAL_TACMAP_GAUSSIAN_SIGMA = 1.5
VISUAL_LOCAL_TACMAP_RAY_DISPLAY_LENGTH_M = 0.002
parser = argparse.ArgumentParser(description="Visualize RL pressure-pad and TacMap observations.")
parser.add_argument("--task", default="BrainCo-Dexsuite-Revo3-Right-Lift-v0")
parser.add_argument("--robot-asset", choices=("right", "bimanual"), default="right")
parser.add_argument("--bimanual-urdf", type=Path, default=BIMANUAL_URDF)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=-1)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--benchmark-steps",
    type=int,
    default=0,
    help=(
        "Measure this many complete Revo3 control/observation steps. A positive value enables "
        "headless benchmark mode and excludes all 2D/UI visualization from the timed region."
    ),
)
parser.add_argument(
    "--benchmark-warmup-steps",
    type=int,
    default=200,
    help="Untimed warm-up steps before --benchmark-steps are measured.",
)
parser.add_argument(
    "--benchmark-output",
    type=Path,
    default=None,
    help="Optional JSON result path for benchmark mode.",
)
parser.add_argument(
    "--benchmark-sync-cuda",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Synchronize CUDA around every measured step for end-to-end latency timing.",
)
parser.add_argument("--viewer-eye", "--viewport-camera-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument(
    "--viewer-lookat",
    "--viewport-camera-target",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
)
parser.add_argument("--print-viewer-camera", action="store_true", default=False)
parser.add_argument("--print-viewer-camera-every", type=int, default=60)
parser.add_argument("--target", choices=("cycle", "tacmap", "pressure_pad"), default="pressure_pad")
parser.add_argument("--focus-finger", choices=FINGER_CHOICES, default="index")
parser.add_argument(
    "--pressure-pad-segment",
    choices=PRESSURE_PAD_SEGMENTS,
    default="mcp",
    help="Pressure target: focus-finger MCP/PIP pad, or the 36-taxel palm pad (palm ignores --focus-finger).",
)
parser.add_argument("--presser", choices=tuple(PRESSER_SPECS), default="square_4")
parser.add_argument("--presser-scale", type=float, default=1.0)
parser.add_argument("--presser-mass", type=float, default=0.2)
parser.add_argument("--presser-contact-offset", type=float, default=None)
parser.add_argument("--presser-rest-offset", type=float, default=None)
parser.add_argument(
    "--pressure-pad-contact-offset",
    type=float,
    default=None,
    help="Override PhysX contact_offset for the 10 pressure-pad MCP/PIP roll links.",
)
parser.add_argument(
    "--pressure-pad-rest-offset",
    type=float,
    default=None,
    help="Override PhysX rest_offset for the 10 pressure-pad MCP/PIP roll links.",
)
parser.add_argument(
    "--presser-body-mode",
    choices=("kinematic", "dynamic_axis"),
    default="dynamic_axis",
    help=(
        "dynamic_axis (default): collision-aware dynamic body constrained to the press normal; "
        "kinematic: direct pose playback that may overlap collision geometry."
    ),
)
parser.add_argument(
    "--presser-axis-max-travel",
    type=float,
    default=0.0,
    help="Maximum dynamic_axis travel in meters. 0 locks hand-driven presses; object-driven presses infer travel from offsets.",
)
parser.add_argument(
    "--presser-axis-max-speed",
    type=float,
    default=0.08,
    help="Max dynamic_axis linear speed in m/s, including force-controlled approach speed.",
)
parser.add_argument(
    "--presser-force-n",
    type=float,
    default=0.0,
    help="Apply a constant force in newtons along the dynamic_axis press direction. 0 disables force control.",
)
parser.add_argument(
    "--presser-force-ramp-steps",
    type=int,
    default=0,
    help="Linearly ramp from 0 to --presser-force-n over this many environment steps. 0 applies it immediately.",
)
presser_collision_group = parser.add_mutually_exclusive_group()
presser_collision_group.add_argument(
    "--enable-presser-collision",
    dest="presser_collision_enabled",
    action="store_true",
    help="Let the scripted presser participate in PhysX contact.",
)
presser_collision_group.add_argument(
    "--disable-presser-collision",
    dest="presser_collision_enabled",
    action="store_false",
    help="Keep the presser visible/queryable for tactile sensors, but remove it from PhysX contact.",
)
parser.set_defaults(presser_collision_enabled=True)
parser.add_argument("--extra-rot-axis", choices=("auto", "none", "x", "y", "z"), default="auto")
parser.add_argument("--extra-rot-deg", type=float, default=None)
parser.add_argument("--flip-axis", choices=("auto", "none", "x", "y", "z"), default="auto")
parser.add_argument("--press-start-offset", type=float, default=0.035)
parser.add_argument("--press-end-offset", type=float, default=0.017)
parser.add_argument("--press-steps", type=int, default=200)
parser.add_argument("--press-slide-distance", type=float, default=0.0)
parser.add_argument("--press-slide-steps", type=int, default=200)
parser.add_argument("--press-slide-axis", type=str, default="+u")
parser.add_argument(
    "--tacmap-press-axis",
    choices=("surface_normal", "camera_optical"),
    default="surface_normal",
    help="Place and move the TacMap presser along the mean surface normal or the fingertip camera principal ray.",
)
parser.add_argument("--tacmap-press-tilt-axis", choices=("none", "+u", "-u", "+v", "-v"), default="none")
parser.add_argument("--tacmap-press-tilt-deg", type=float, default=0.0)
parser.add_argument(
    "--press-motion-actor",
    choices=("object", "hand"),
    default="object",
    help="object: move the presser into the hand; hand: keep the presser fixed and press with joint targets.",
)
parser.add_argument(
    "--press-finger-joint",
    action="append",
    default=None,
    help="Joint to drive for --press-motion-actor hand. Defaults to the focus finger MCP/PIP roll joint.",
)
parser.add_argument("--press-finger-start-rad", type=float, default=0.0)
parser.add_argument("--press-finger-end-rad", type=float, default=0.2)
parser.add_argument("--pressure-pad-press-start-offset", type=float, default=0.025)
parser.add_argument("--pressure-pad-press-end-offset", type=float, default=-0.005)
parser.add_argument("--pressure-pad-press-steps", type=int, default=100)
parser.add_argument(
    "--robot-hold-mode",
    choices=("target", "sharpa", "root", "free_focus", "hard", "none"),
    default="target",
    help=(
        "target: keep joint position targets but let contact move the robot; "
        "sharpa: keep the robot root fixed and drive joint position targets; "
        "root: keep only the robot root fixed and leave joints free; "
        "free_focus: keep root and non-focus joints fixed while focus-finger targets follow current joint positions; "
        "hard: rewrite joint/root state every physics step; none: do not hold robot joints."
    ),
)
parser.add_argument(
    "--sharpa-finger-target-gain",
    type=float,
    default=0.05,
    help="Extra Sharpa-style correction gain for the focus finger when --robot-hold-mode sharpa.",
)
parser.add_argument("--cycle-hold-steps", type=int, default=30)
parser.add_argument("--pressure-scale", type=int, default=20)
parser.add_argument("--pressure-gamma", type=float, default=1.0)
parser.add_argument(
    "--pressure-display-diffusion",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Apply CPU-only Gaussian diffusion to displayed pressure taxels. "
        "The sensor tensor and RL observation remain unchanged."
    ),
)
parser.add_argument(
    "--pressure-display-diffusion-sigma-mm",
    type=float,
    default=3.0,
    help="Physical Gaussian sigma in millimeters for display-only pressure diffusion.",
)
parser.add_argument(
    "--pressure-display-diffusion-blend",
    type=float,
    default=0.7,
    help="Display-only blend between raw and diffused pressure in [0, 1].",
)
parser.add_argument(
    "--pressure-display-diffusion-radius-sigma",
    type=float,
    default=3.0,
    help="Display-only Gaussian cutoff radius as a multiple of sigma.",
)
parser.add_argument(
    "--pressure-display-diffusion-normal-power",
    type=float,
    default=1.0,
    help="Exponent applied to taxel-normal alignment in the display-only Gaussian kernel.",
)
parser.add_argument("--tacmap-scale", type=int, default=16)
parser.add_argument("--tacmap-gamma", type=float, default=1.0)
parser.add_argument("--tacmap-display-max-mm", type=float, default=1.0)
parser.add_argument("--tacmap-display-rows", type=int, default=240)
parser.add_argument("--tacmap-display-cols", type=int, default=320)
parser.add_argument("--tacmap-resize-mode", choices=("nearest", "bilinear", "surface"), default="bilinear")
taxim_rgb_group = parser.add_mutually_exclusive_group()
taxim_rgb_group.add_argument(
    "--show-taxim-rgb",
    dest="show_taxim_rgb",
    action="store_true",
    help="Append Taxim RGB rendered from the float32 240x320 local TacMap penetration raster.",
)
taxim_rgb_group.add_argument(
    "--hide-taxim-rgb",
    dest="show_taxim_rgb",
    action="store_false",
    help="Disable the additional Taxim RGB visualization pane.",
)
parser.set_defaults(show_taxim_rgb=True)
parser.add_argument(
    "--taxim-rgb-depth-scale",
    type=float,
    default=1.0,
    help="Depth scale shared by the RL Taxim RGB observation and its visualization.",
)
parser.add_argument(
    "--taxim-rgb-with-shadow",
    action="store_true",
    default=False,
    help="Enable shadows in both the RL Taxim RGB observation and its visualization.",
)
parser.add_argument(
    "--taxim-rgb-device",
    type=str,
    default=None,
    help="Optional renderer device shared by the RL Taxim RGB observation and its visualization.",
)
parser.add_argument(
    "--taxim-rgb-background",
    choices=("marker", "markerless", "taxim"),
    default="marker",
    help="Background shared by the RL Taxim RGB observation and its visualization.",
)
parser.add_argument(
    "--tacmap-local-refinement",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        f"Display only: use {VISUAL_LOCAL_TACMAP_RAY_COUNT} adaptive object rays around the focus contact. "
        "A dense rubber-surface reference is ray-cast once at startup; RL observations remain unchanged."
    ),
)
parser.add_argument(
    "--tacmap-local-contact-threshold-mm",
    type=float,
    default=0.02,
    help="Depth threshold used to choose the focus-contact connected component for local refinement.",
)
parser.add_argument(
    "--tacmap-local-roi-margin-mm",
    type=float,
    default=1.0,
    help=f"Camera-XY metric margin in millimetres around the focus-contact box before placing the {VISUAL_LOCAL_TACMAP_RAY_COUNT} local rays.",
)
parser.add_argument(
    "--tacmap-local-roi-guard-mm",
    type=float,
    default=0.0,
    help="Optional camera-XY tolerance before re-indexing the local rays; zero follows every 32x24 ROI change.",
)
parser.add_argument(
    "--tacmap-local-roi-stable-frames",
    type=int,
    default=1,
    help="Require a new local-ray ROI for this many environment updates before moving all local rays.",
)
local_tacmap_point_group = parser.add_mutually_exclusive_group()
local_tacmap_point_group.add_argument(
    "--show-local-tacmap-points",
    dest="show_local_tacmap_points",
    action="store_true",
    help=(
        f"Show the {VISUAL_LOCAL_TACMAP_RAY_COUNT} local-ray sample positions on the calibrated rubber "
        "reference surface."
    ),
)
local_tacmap_point_group.add_argument(
    "--hide-local-tacmap-points",
    dest="show_local_tacmap_points",
    action="store_false",
    help=f"Hide the {VISUAL_LOCAL_TACMAP_RAY_COUNT} display-only local TacMap sample points.",
)
parser.set_defaults(show_local_tacmap_points=True)
parser.add_argument("--local-tacmap-point-radius", type=float, default=0.00008)
parser.add_argument(
    "--no-tacmap-marker-depth-fusion",
    dest="tacmap_marker_depth_fusion",
    action="store_false",
    default=True,
    help=(
        "Disable display-only joint interpolation of regular TacMap rays and independent calibrated Marker-ray depths."
    ),
)
parser.add_argument(
    "--tacmap-display-rot90",
    type=int,
    choices=(0, 1, 2, 3),
    default=0,
    help="Display-only TacMap rotation in 90 degree counter-clockwise steps. RL observation tensors are unchanged.",
)
parser.add_argument(
    "--tacmap-display-flip-x",
    action="store_true",
    default=False,
    help="Display-only TacMap horizontal flip after rotation. RL observation tensors are unchanged.",
)
parser.add_argument(
    "--tacmap-display-flip-y",
    action="store_true",
    default=False,
    help="Display-only TacMap vertical flip after rotation. RL observation tensors are unchanged.",
)
parser.add_argument(
    "--tacmap-contact-shell",
    type=float,
    default=-1.0,
    help="Virtual compliant shell in meters for non-penetrating TacMap contact; negative disables it.",
)
tacmap_all_group = parser.add_mutually_exclusive_group()
tacmap_all_group.add_argument("--show-all-tacmap-fingers", dest="show_all_tacmap_fingers", action="store_true")
tacmap_all_group.add_argument("--hide-all-tacmap-fingers", dest="show_all_tacmap_fingers", action="store_false")
tacmap_contact_group = parser.add_mutually_exclusive_group()
tacmap_contact_group.add_argument("--show-tacmap-contact-window", dest="show_tacmap_contact_window", action="store_true")
tacmap_contact_group.add_argument("--hide-tacmap-contact-window", dest="show_tacmap_contact_window", action="store_false")
parser.set_defaults(show_all_tacmap_fingers=True, show_tacmap_contact_window=False)
surface_debug_group = parser.add_mutually_exclusive_group()
surface_debug_group.add_argument("--show-surface-debug", dest="show_surface_debug", action="store_true")
surface_debug_group.add_argument("--hide-surface-debug", dest="show_surface_debug", action="store_false")
parser.set_defaults(show_surface_debug=True)
parser.add_argument("--surface-debug-all-fingers", action="store_true", default=False)
parser.add_argument("--surface-debug-point-max", type=int, default=12000)
parser.add_argument("--surface-debug-normal-stride", type=int, default=8)
parser.add_argument("--surface-debug-normal-length", type=float, default=0.004)
parser.add_argument("--surface-debug-point-radius", type=float, default=0.0008)
parser.add_argument("--surface-debug-ray-width", type=float, default=0.00006)
selected_surface_point_group = parser.add_mutually_exclusive_group()
selected_surface_point_group.add_argument(
    "--show-selected-surface-points",
    dest="show_selected_surface_points",
    action="store_true",
)
selected_surface_point_group.add_argument(
    "--hide-selected-surface-points",
    dest="show_selected_surface_points",
    action="store_false",
)
parser.set_defaults(show_selected_surface_points=False)
parser.add_argument("--selected-surface-point-radius", type=float, default=0.00025)
parser.add_argument("--show-native-tacmap-debug", action="store_true", default=False)
parser.add_argument("--debug-sensors", action="store_true", default=False)
parser.add_argument(
    "--pause-on-tacmap-onset",
    action="store_true",
    default=False,
    help="Debug only: pause physics when the focus TacMap first becomes nonzero.",
)
parser.add_argument("--tacmap-onset-threshold-mm", type=float, default=0.0)
hydroshear_marker_group = parser.add_mutually_exclusive_group()
hydroshear_marker_group.add_argument("--show-hydroshear-marker", dest="show_hydroshear_marker", action="store_true")
hydroshear_marker_group.add_argument("--hide-hydroshear-marker", dest="show_hydroshear_marker", action="store_false")
parser.set_defaults(show_hydroshear_marker=True)
parser.add_argument(
    "--hydroshear-marker-motion-scale",
    type=float,
    default=1.0,
    help="Display-only scale for HydroShear marker displacement vectors. RL observations remain unchanged.",
)
calibrated_marker_group = parser.add_mutually_exclusive_group()
calibrated_marker_group.add_argument(
    "--show-calibrated-marker-points", dest="show_calibrated_marker_points", action="store_true"
)
calibrated_marker_group.add_argument(
    "--hide-calibrated-marker-points", dest="show_calibrated_marker_points", action="store_false"
)
parser.set_defaults(show_calibrated_marker_points=True)
camera_frame_group = parser.add_mutually_exclusive_group()
camera_frame_group.add_argument("--show-camera-frame", dest="show_camera_frame", action="store_true")
camera_frame_group.add_argument("--hide-camera-frame", dest="show_camera_frame", action="store_false")
parser.set_defaults(show_camera_frame=True)
parser.add_argument("--camera-frame-axis-length", type=float, default=0.008)
parser.add_argument("--camera-frame-axis-width", type=float, default=0.00012)
camera_visible_group = parser.add_mutually_exclusive_group()
camera_visible_group.add_argument(
    "--show-camera-visible-surface", dest="show_camera_visible_surface", action="store_true"
)
camera_visible_group.add_argument(
    "--hide-camera-visible-surface", dest="show_camera_visible_surface", action="store_false"
)
parser.set_defaults(show_camera_visible_surface=True)
parser.add_argument("--camera-visible-surface-samples", type=int, default=3000)
parser.add_argument("--camera-visible-boundary-points", type=int, default=128)
parser.add_argument("--camera-visible-point-radius", type=float, default=0.00015)
hydroshear_roi_group = parser.add_mutually_exclusive_group()
hydroshear_roi_group.add_argument("--show-hydroshear-roi", dest="show_hydroshear_roi", action="store_true")
hydroshear_roi_group.add_argument("--hide-hydroshear-roi", dest="show_hydroshear_roi", action="store_false")
parser.set_defaults(show_hydroshear_roi=True)
parser.add_argument("--hydroshear-roi-line-width", type=float, default=0.00008)
parser.add_argument("--hydroshear-roi-grid-stride", type=int, default=4)
parser.add_argument("--hydroshear-roi-point-radius", type=float, default=0.00032)
parser.add_argument("--hydroshear-tacmap-point-radius", type=float, default=0.00016)
parser.add_argument("--hydroshear-marker-point-radius", type=float, default=0.00055)
parser.add_argument("--hydroshear-marker-normal-length", type=float, default=0.003)
parser.add_argument("--hydroshear-marker-normal-width", type=float, default=0.00008)
parser.add_argument("--hydroshear-marker-axis-length", type=float, default=0.002)
parser.add_argument("--hydroshear-marker-axis-width", type=float, default=0.00006)
parser.add_argument(
    "--visual-update-interval",
    type=int,
    default=2,
    help=(
        "Refresh USD and 2D debug output every N control steps while RL observations still update every step. "
        "The default 2 gives a 30 Hz visual stream for this 60 Hz task."
    ),
)
parser.add_argument(
    "--focus-visuotactile-only",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Show only the focus finger's Depth, Taxim RGB, and HydroShear Marker panes. "
        "This mode also hides pressure panes and every 3D surface-debug overlay. "
        "It is enabled by default for --target tacmap and can be disabled with "
        "--no-focus-visuotactile-only."
    ),
)
parser.add_argument(
    "--focus-pressure-only",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Show only the pressure map selected by --focus-finger and --pressure-pad-segment. "
        "This mode also hides all visuotactile panes and 3D surface-debug overlays. "
        "It is enabled by default for --target pressure_pad and can be disabled with "
        "--no-focus-pressure-only."
    ),
)
parser.add_argument(
    "--whole-hand-pressure-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Show all 11 pressure regions (285 taxels) as one pressure-only panel. "
        "This hides every Depth, RGB, Marker, and 3D surface-debug view."
    ),
)
parser.add_argument("--no-local-ui", action="store_true", default=False)
parser.add_argument("--show-cv", action="store_true", default=False)
parser.add_argument("--record-video", type=str, default=None, help="Write the combined observation panel to an MP4 file.")
parser.add_argument("--save-final-image", type=str, default=None, help="Write the final observation panel to an image file.")
parser.add_argument("--record-fps", type=float, default=0.0, help="Recording FPS; 0 uses the environment step rate.")
parser.add_argument(
    "--record-side-view",
    action="store_true",
    default=False,
    help="Prepend a square simulator side-view pane to the local UI and recorded MP4.",
)
parser.add_argument("--side-view-resolution", type=int, default=320)
parser.add_argument("--side-view-distance", type=float, default=0.10, help="Side camera distance from the contact center in metres.")
parser.add_argument("--side-view-elevation", type=float, default=0.035, help="Side camera elevation above the contact center in metres.")
parser.add_argument("--print-every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.focus_visuotactile_only is None:
    args_cli.focus_visuotactile_only = str(args_cli.target) == "tacmap" and not bool(
        args_cli.whole_hand_pressure_only
    )
if args_cli.focus_pressure_only is None:
    args_cli.focus_pressure_only = str(args_cli.target) == "pressure_pad" and not bool(
        args_cli.whole_hand_pressure_only
    )
if sum(
    bool(value)
    for value in (
        args_cli.focus_visuotactile_only,
        args_cli.focus_pressure_only,
        args_cli.whole_hand_pressure_only,
    )
) > 1:
    parser.error(
        "--focus-visuotactile-only, --focus-pressure-only, and --whole-hand-pressure-only are mutually exclusive"
    )
if bool(args_cli.focus_visuotactile_only):
    # A compact observation-only view: no other fingers, pressure panes, or
    # surface/ray/marker debug geometry in the Isaac scene.
    args_cli.show_all_tacmap_fingers = False
    args_cli.show_tacmap_contact_window = False
    args_cli.show_surface_debug = False
    args_cli.show_selected_surface_points = False
    args_cli.show_local_tacmap_points = False
    args_cli.show_calibrated_marker_points = False
    args_cli.show_camera_frame = False
    args_cli.show_camera_visible_surface = False
    args_cli.show_hydroshear_roi = False
    args_cli.show_native_tacmap_debug = False
if bool(args_cli.focus_pressure_only):
    # A single-pad observation view. Skip every visuotactile pane and its
    # associated display-only geometry while preserving all sensor tensors.
    args_cli.show_taxim_rgb = False
    args_cli.show_hydroshear_marker = False
    args_cli.show_all_tacmap_fingers = False
    args_cli.show_tacmap_contact_window = False
    args_cli.show_surface_debug = False
    args_cli.show_selected_surface_points = False
    args_cli.show_local_tacmap_points = False
    args_cli.show_calibrated_marker_points = False
    args_cli.show_camera_frame = False
    args_cli.show_camera_visible_surface = False
    args_cli.show_hydroshear_roi = False
    args_cli.show_native_tacmap_debug = False
if bool(args_cli.whole_hand_pressure_only):
    # Keep all 11 pads in the canonical 285-D observation order, but remove
    # every non-pressure visualization and display-only 3D overlay.
    args_cli.show_taxim_rgb = False
    args_cli.show_hydroshear_marker = False
    args_cli.show_all_tacmap_fingers = False
    args_cli.show_tacmap_contact_window = False
    args_cli.show_surface_debug = False
    args_cli.show_selected_surface_points = False
    args_cli.show_local_tacmap_points = False
    args_cli.show_calibrated_marker_points = False
    args_cli.show_camera_frame = False
    args_cli.show_camera_visible_surface = False
    args_cli.show_hydroshear_roi = False
    args_cli.show_native_tacmap_debug = False
if int(args_cli.benchmark_steps) > 0:
    # Benchmark the complete policy observation path, not the display-only
    # rendering/readback path. These overrides happen before AppLauncher so a
    # benchmark is headless even when the caller forgets ``--headless``.
    args_cli.headless = True
    args_cli.no_local_ui = True
    args_cli.real_time = False
    args_cli.show_cv = False
    args_cli.show_surface_debug = False
    args_cli.show_selected_surface_points = False
    args_cli.show_local_tacmap_points = False
    args_cli.show_calibrated_marker_points = False
    args_cli.show_camera_frame = False
    args_cli.show_camera_visible_surface = False
    args_cli.show_hydroshear_roi = False
    args_cli.show_native_tacmap_debug = False
    args_cli.show_hydroshear_marker = False
    args_cli.show_taxim_rgb = False
    args_cli.show_tacmap_contact_window = False
    args_cli.record_video = None
    args_cli.record_side_view = False
    args_cli.save_final_image = None
if bool(args_cli.record_side_view):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject  # noqa: E402
from isaaclab.sim.converters import UrdfConverterCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import BrainCo_DexHand  # noqa: F401,E402
from BrainCo_DexHand.force_map import load_pressure_pad_specs_from_urdf  # noqa: E402
from BrainCo_DexHand.tasks.manager_based.dexsuite.config.Revo3.dexsuite_revo3_env_cfg_grasp import (  # noqa: E402
    TIANJI_PRESSURE_SENSOR_NAMES,
    TIANJI_PRESSURE_PAD_LINK_ORDER,
    TIANJI_PRESSUREPAD_URDF,
    TIANJI_RL_PRESSURE_DIFFUSION_BLEND,
    TIANJI_RL_PRESSURE_DIFFUSION_ENABLED,
    TIANJI_RL_PRESSURE_DIFFUSION_SIGMA_M,
    TIANJI_HYDROSHEAR_BASELINE_MARKER_COLS,
    TIANJI_HYDROSHEAR_BASELINE_MARKER_ROWS,
    TIANJI_HYDROSHEAR_BASELINE_RAY_COLS,
    TIANJI_HYDROSHEAR_BASELINE_RAY_ROWS,
    TIANJI_HYDROSHEAR_BASELINE_RENDER_COLS,
    TIANJI_HYDROSHEAR_BASELINE_RENDER_ROWS,
    TIANJI_FOTS_BASELINE_CONTACT_THRESHOLD_MM,
    TIANJI_FOTS_BASELINE_LAMB,
    TIANJI_FOTS_BASELINE_MARKER_COLS,
    TIANJI_FOTS_BASELINE_MARKER_MARGIN_X,
    TIANJI_FOTS_BASELINE_MARKER_MARGIN_Y,
    TIANJI_FOTS_BASELINE_MARKER_ROWS,
    TIANJI_FOTS_BASELINE_MM2PIX,
    TIANJI_FOTS_BASELINE_RAY_COLS,
    TIANJI_FOTS_BASELINE_RAY_ROWS,
    TIANJI_FOTS_BASELINE_RENDER_COLS,
    TIANJI_FOTS_BASELINE_RENDER_ROWS,
    TIANJI_TACMAP_LINK_ORDER,
    TIANJI_TACMAP_LINK_SURFACE_DEFAULTS,
    TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M,
    TIANJI_TACMAP_OBJECT_SENSOR_NAMES,
    TIANJI_TACMAP_RL_COLS,
    TIANJI_TACMAP_RL_ROWS,
    TIANJI_TACMAP_SURFACE_SENSOR_NAMES,
    TIANJI_TACSL_BASELINE_OUTPUT_COLS,
    TIANJI_TACSL_BASELINE_OUTPUT_ROWS,
    TIANJI_TACSL_BASELINE_RAY_COLS,
    TIANJI_TACSL_BASELINE_RAY_ROWS,
    _make_tacmap_link_surface_cfg,
    _mean_tacmap_normal_direction,
    _tacmap_asset_paths,
)
from BrainCo_DexHand.tasks.manager_based.dexsuite.mdp.observations import (  # noqa: E402
    _camera_plane_ray_grid_from_rectangle,
    _load_camera_ray_rectangles,
    ours_rl_taxim_rgb_obs,
)


def quat_mul_tuple(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate_np(q: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    w, x, y, z = q
    rot = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return np.asarray(vecs, dtype=np.float64) @ rot.T


def presser_bbox_corners_l() -> np.ndarray:
    if args_cli.presser != "square_4":
        return np.zeros((0, 3), dtype=np.float64)
    mins = np.asarray(SQUARE_4_MESH_BBOX_L[0], dtype=np.float64) * float(args_cli.presser_scale)
    maxs = np.asarray(SQUARE_4_MESH_BBOX_L[1], dtype=np.float64) * float(args_cli.presser_scale)
    return np.asarray(
        [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
        dtype=np.float64,
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
            vec = xyz_axis(ray_axis) if norm < 1.0e-12 else vec / norm
    else:
        vec = xyz_axis(axis)
        sign = 1.0
    vec = sign * vec
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def normalize_np(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1.0e-8:
        vec = np.asarray(fallback, dtype=np.float32)
        norm = max(float(np.linalg.norm(vec)), 1.0e-8)
    return vec / norm


def quat_from_two_vectors_np(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize_np(np.asarray(src, dtype=np.float32), np.asarray((1.0, 0.0, 0.0), dtype=np.float32)).astype(np.float64)
    dst = normalize_np(np.asarray(dst, dtype=np.float32), np.asarray((1.0, 0.0, 0.0), dtype=np.float32)).astype(np.float64)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    if dot < -1.0 + 1.0e-8:
        axis = np.cross(src, np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1.0e-8:
            axis = np.cross(src, np.asarray((0.0, 1.0, 0.0), dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-8)
        return np.asarray((0.0, axis[0], axis[1], axis[2]), dtype=np.float32)
    axis = np.cross(src, dst)
    quat = np.asarray((1.0 + dot, axis[0], axis[1], axis[2]), dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-8)
    return quat.astype(np.float32)


def tacmap_press_tilt(normal_l: np.ndarray, *, params: dict) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize_np(np.asarray(normal_l, dtype=np.float32), np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    axis = str(args_cli.tacmap_press_tilt_axis).lower()
    degrees = float(args_cli.tacmap_press_tilt_deg)
    if axis == "none" or abs(degrees) <= 1.0e-12:
        return normal, np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    direction = np.asarray(
        axis_to_vector(
            axis,
            grid_u_axis=str(params["grid_u_axis"]),
            grid_v_axis=str(params["grid_v_axis"]),
            ray_axis=str(params["ray_axis"]),
            ray_direction=tuple(float(v) for v in normal),
        ),
        dtype=np.float32,
    )
    tangent = direction - normal * float(np.dot(direction, normal))
    tangent = normalize_np(tangent, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
    radians = math.radians(degrees)
    tilted = normalize_np(normal * math.cos(radians) + tangent * math.sin(radians), normal)
    return tilted, quat_from_two_vectors_np(normal, tilted)


def pressure_pad_link_name(finger: str, segment: str) -> str:
    if str(segment) == "palm":
        return "right_hand_rubber_link"
    stem = PRESSURE_PAD_FINGER_STEMS[finger]
    return f"right_{stem}{segment}_roll_touch_link"


def default_press_joint_names(finger: str, phase: str) -> tuple[str, ...]:
    if args_cli.press_finger_joint:
        return tuple(str(name).strip() for name in args_cli.press_finger_joint if str(name).strip())
    stem = FINGER_ROLL_JOINT_STEMS[str(finger)]
    segment = "pip" if phase.startswith("pressure_pad") and args_cli.pressure_pad_segment == "pip" else "mcp"
    return (f"right_{stem}{segment}_roll_joint",)


def pressure_pad_target(link_name: str) -> tuple[str, np.ndarray, np.ndarray]:
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    specs_by_link = {str(spec.link_name): spec for spec in specs}
    spec = specs_by_link.get(link_name)
    if spec is None:
        raise RuntimeError(f"Pressure pad link {link_name!r} is not declared in {TIANJI_PRESSUREPAD_URDF}.")
    taxel_map = spec.to_taxel_map()
    points_l = np.asarray(taxel_map.points_l, dtype=np.float32)
    normals_l = np.asarray(taxel_map.normals_l, dtype=np.float32)
    fallback = np.zeros(3, dtype=np.float32)
    fallback[int(spec.normal_axis)] = 1.0 if float(spec.normal_sign) >= 0.0 else -1.0
    center_l = np.mean(points_l, axis=0).astype(np.float32)
    if str(spec.link_name) == "right_hand_rubber_link":
        # The palm is an even 6x6 layout, so its geometric centroid lies in the
        # empty gap between four taxels. Align a narrow probe with the nearest
        # actual central taxel while retaining the complete 36-taxel output.
        center_index = int(np.argmin(np.sum((points_l - center_l[None, :]) ** 2, axis=1)))
        center_l = points_l[center_index].astype(np.float32, copy=True)
        normal_l = normalize_np(normals_l[center_index], fallback)
    else:
        normal_l = normalize_np(np.mean(normals_l, axis=0).astype(np.float32), fallback)
    return (
        str(spec.link_name),
        center_l,
        normal_l,
    )


def camera_optical_tacmap_target(finger: str, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    import xml.etree.ElementTree as ET

    from scripts.project_vitai_markers_to_mesh import (
        FINGERS,
        element_transform,
        load_surface_triangles,
        ray_triangle_nearest,
    )

    root = ET.parse(VITAI_RIGHT_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}
    config = FINGERS[finger]
    if str(config["surface_link"]) != link_name:
        raise ValueError(f"Camera surface link {config['surface_link']!r} does not match TacMap link {link_name!r}")

    camera_xyz_parent, camera_rotation_parent = element_transform(joints[config["camera_joint"]])
    surface_xyz_parent, surface_rotation_parent = element_transform(joints[config["surface_joint"]])
    camera_origin_l = surface_rotation_parent.T @ (camera_xyz_parent - surface_xyz_parent)
    camera_rotation_l = surface_rotation_parent.T @ camera_rotation_parent
    optical_axis_l = normalize_np(
        camera_rotation_l @ np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
    )

    _, triangles, _ = load_surface_triangles(VITAI_RIGHT_URDF, links[config["surface_link"]])
    first_hit = ray_triangle_nearest(camera_origin_l, optical_axis_l, triangles)
    if first_hit is None:
        raise RuntimeError(f"{finger} camera principal ray does not intersect {link_name}")
    distance = float(first_hit[1])
    outer_origin = camera_origin_l + (distance + 1.0e-5) * optical_axis_l
    outer_hit = ray_triangle_nearest(outer_origin, optical_axis_l, triangles)
    if outer_hit is not None:
        distance += 1.0e-5 + float(outer_hit[1])
    return (
        (camera_origin_l + distance * optical_axis_l).astype(np.float32),
        optical_axis_l.astype(np.float32),
    )


def tacmap_target(finger: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    link_name = TIANJI_TACMAP_LINK_ORDER[FINGER_CHOICES.index(finger)]
    params = TIANJI_TACMAP_LINK_SURFACE_DEFAULTS[link_name]
    _points_npy, normals_npy = _tacmap_asset_paths(link_name)
    mean_normal = _mean_tacmap_normal_direction(str(normals_npy))
    if mean_normal is None:
        mean_normal = axis_to_vector(str(params["ray_axis"]))
    slide_axis_l = axis_to_vector(
        args_cli.press_slide_axis,
        grid_u_axis=str(params["grid_u_axis"]),
        grid_v_axis=str(params["grid_v_axis"]),
        ray_axis=str(params["ray_axis"]),
        ray_direction=mean_normal,
    )
    center_l = np.asarray(params["grid_center"], dtype=np.float32)
    if str(args_cli.tacmap_press_axis) == "camera_optical":
        center_l, normal_l = camera_optical_tacmap_target(finger, link_name)
        tilt_quat_l = quat_from_two_vectors_np(np.asarray(mean_normal, dtype=np.float32), normal_l)
    else:
        normal_l, tilt_quat_l = tacmap_press_tilt(np.asarray(mean_normal, dtype=np.float32), params=params)
    return (
        link_name,
        center_l,
        normal_l,
        normalize_np(np.asarray(slide_axis_l, dtype=np.float32), np.asarray((0.0, 1.0, 0.0), dtype=np.float32)),
        tilt_quat_l,
    )


def presser_quat_l() -> tuple[float, float, float, float]:
    spec = PRESSER_SPECS[args_cli.presser]
    extra_axis = str(spec["default_extra_rot_axis"]) if args_cli.extra_rot_axis == "auto" else args_cli.extra_rot_axis
    extra_deg = float(spec["default_extra_rot_deg"]) if args_cli.extra_rot_deg is None else float(args_cli.extra_rot_deg)
    flip_axis = str(spec["default_flip_axis"]) if args_cli.flip_axis == "auto" else args_cli.flip_axis
    return quat_mul_tuple(
        quat_mul_tuple(tuple(float(v) for v in spec["rot"]), axis_angle_to_quat(extra_axis, extra_deg)),
        flip_axis_to_quat(flip_axis),
    )


class RobotHold:
    def __init__(
        self,
        env,
        *,
        focus_finger: str,
        press_joint_names: tuple[str, ...],
        sharpa_finger_target_gain: float,
    ) -> None:
        self.env = env
        self.robot: Articulation = env.scene["robot"]
        self.root_pose = torch.cat((self.robot.data.root_pos_w.clone(), self.robot.data.root_quat_w.clone()), dim=-1)
        self.root_vel = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32)
        self.joint_pos = self.robot.data.joint_pos.clone()
        self.joint_vel = torch.zeros_like(self.robot.data.joint_vel)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.sharpa_finger_target_gain = float(sharpa_finger_target_gain)
        needles = FINGER_JOINT_NAME_NEEDLES.get(str(focus_finger), (f"right_{focus_finger}",))
        self.sharpa_finger_joint_ids = [
            i for i, name in enumerate(self.robot.joint_names) if any(needle in str(name) for needle in needles)
        ]
        if self.sharpa_finger_joint_ids:
            names = [str(self.robot.joint_names[i]) for i in self.sharpa_finger_joint_ids]
            print(
                "[INFO] Sharpa-style focus finger target correction: "
                f"finger={focus_finger}, gain={self.sharpa_finger_target_gain}, joints={names}",
                flush=True,
            )
        else:
            print(f"[WARN] No focus-finger joints matched for Sharpa-style hold: finger={focus_finger}", flush=True)

        joint_names = [str(name) for name in self.robot.joint_names]
        self.press_joint_ids: list[int] = []
        self.press_joint_names: list[str] = []
        for requested in press_joint_names:
            matches = [i for i, name in enumerate(joint_names) if name == requested]
            if not matches:
                matches = [i for i, name in enumerate(joint_names) if name.endswith(requested)]
            if not matches:
                raise RuntimeError(f"Press finger joint {requested!r} not found in robot joints: {joint_names}")
            joint_id = int(matches[0])
            if joint_id not in self.press_joint_ids:
                self.press_joint_ids.append(joint_id)
                self.press_joint_names.append(joint_names[joint_id])
        print(
            "[INFO] Sharpa-style press joints: "
            f"{self.press_joint_names}, "
            f"target={float(args_cli.press_finger_start_rad):g}->{float(args_cli.press_finger_end_rad):g} rad",
            flush=True,
        )

    def sharpa_joint_target(self) -> torch.Tensor:
        target = self.default_joint_pos.clone()
        if self.sharpa_finger_joint_ids and self.sharpa_finger_target_gain != 0.0:
            ids = self.sharpa_finger_joint_ids
            current = self.robot.data.joint_pos[:, ids]
            default = self.default_joint_pos[:, ids]
            target[:, ids] = default - self.sharpa_finger_target_gain * (current - default)
        return target

    def press_joint_target(self, alpha: float) -> torch.Tensor:
        target = self.joint_pos.clone()
        if self.press_joint_ids:
            start = float(args_cli.press_finger_start_rad)
            end = float(args_cli.press_finger_end_rad)
            target_value = start + float(alpha) * (end - start)
            target[:, self.press_joint_ids] = target_value
        return target

    def focus_free_joint_target(self) -> torch.Tensor:
        target = self.joint_pos.clone()
        if self.sharpa_finger_joint_ids:
            ids = self.sharpa_finger_joint_ids
            target[:, ids] = self.robot.data.joint_pos[:, ids]
        return target

    def write_root(self) -> None:
        self.robot.write_root_pose_to_sim(self.root_pose)
        self.robot.write_root_velocity_to_sim(self.root_vel)
        self.robot.update(0.0)

    def write(self, mode: str, *, joint_target: torch.Tensor | None = None) -> None:
        mode = str(mode).lower()
        if mode == "none":
            return
        if mode == "root":
            self.write_root()
            return
        target = self.joint_pos if joint_target is None else joint_target
        if mode == "free_focus":
            self.write_root()
            self.robot.set_joint_position_target(self.focus_free_joint_target())
            self.robot.write_data_to_sim()
            return
        if mode == "sharpa":
            self.write_root()
            if joint_target is None:
                target = self.sharpa_joint_target()
            self.robot.set_joint_position_target(target)
            self.robot.write_data_to_sim()
            return
        if mode == "target":
            self.robot.set_joint_position_target(target)
            self.robot.write_data_to_sim()
            return
        self.robot.write_root_pose_to_sim(self.root_pose)
        self.robot.write_root_velocity_to_sim(self.root_vel)
        self.robot.write_joint_state_to_sim(target, self.joint_vel)
        self.robot.update(0.0)
        self.robot.set_joint_position_target(target)
        self.robot.write_data_to_sim()


def body_index(robot: Articulation, link_name: str) -> int:
    matches = [i for i, name in enumerate(robot.body_names) if str(name) == str(link_name)]
    if not matches:
        matches = [i for i, name in enumerate(robot.body_names) if str(name).endswith(str(link_name))]
    if not matches:
        raise RuntimeError(f"Body {link_name!r} not found in robot body names: {list(robot.body_names)}")
    return int(matches[0])


def scripted_target(step: int) -> tuple[str, int, float]:
    tacmap_len = max(1, int(args_cli.press_steps)) + max(0, int(args_cli.press_slide_steps))
    pressure_len = max(1, int(args_cli.pressure_pad_press_steps))
    hold = max(0, int(args_cli.cycle_hold_steps))
    if args_cli.target == "tacmap":
        return "tacmap", step, min(float(step) / float(max(1, int(args_cli.press_steps) - 1)), 1.0)
    if args_cli.target == "pressure_pad":
        return "pressure_pad", step, min(float(step) / float(max(1, pressure_len - 1)), 1.0)

    period = tacmap_len + hold + pressure_len + hold
    local_step = step % max(1, period)
    if local_step < tacmap_len:
        return "tacmap", local_step, min(float(local_step) / float(max(1, int(args_cli.press_steps) - 1)), 1.0)
    local_step -= tacmap_len + hold
    if local_step < 0:
        return "tacmap_hold", tacmap_len - 1, 1.0
    if local_step < pressure_len:
        return "pressure_pad", local_step, min(float(local_step) / float(max(1, pressure_len - 1)), 1.0)
    return "pressure_pad_hold", pressure_len - 1, 1.0


def write_object_pose(
    env,
    link_name: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
    quat_l_np: np.ndarray,
    *,
    offset_m: float,
    slide_l_np: np.ndarray | None = None,
) -> None:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    link_idx = body_index(robot, link_name)
    link_state = robot.data.body_link_state_w[:, link_idx, :7]
    link_pos_w = link_state[:, :3]
    link_quat_w = link_state[:, 3:7]
    center_l = torch.as_tensor(center_l_np, device=env.device, dtype=torch.float32)
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32)
    slide_l = (
        torch.zeros(3, device=env.device, dtype=torch.float32)
        if slide_l_np is None
        else torch.as_tensor(slide_l_np, device=env.device, dtype=torch.float32)
    )
    object_pos_l = center_l + normal_l * float(offset_m) + slide_l
    object_pos_w = link_pos_w + quat_apply(link_quat_w, object_pos_l.unsqueeze(0).expand(env.num_envs, -1))
    quat_l = torch.as_tensor(quat_l_np, device=env.device, dtype=torch.float32).unsqueeze(0).expand(env.num_envs, -1)
    object_quat_w = quat_mul(link_quat_w, quat_l)
    obj.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32))


def dynamic_axis_enabled() -> bool:
    return str(args_cli.presser_body_mode).lower() == "dynamic_axis"


def presser_force_enabled() -> bool:
    return dynamic_axis_enabled() and float(args_cli.presser_force_n) > 0.0


def ensure_fixed_presser_pose(
    env,
    link_name: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
    quat_l_np: np.ndarray,
    *,
    offset_m: float,
    key_base: tuple[str, str],
    force_reset: bool,
) -> None:
    global_step = int(getattr(env, "_rl_tactile_viz_global_step", -1))
    reset_step = getattr(env, "_rl_fixed_presser_reset_step", None)
    if getattr(env, "_rl_fixed_presser_key_base", None) == key_base and not (force_reset and reset_step != global_step):
        return
    write_object_pose(env, link_name, center_l_np, normal_l_np, quat_l_np, offset_m=offset_m)
    robot: Articulation = env.scene["robot"]
    link_idx = body_index(robot, link_name)
    link_quat_w = robot.data.body_link_state_w[:, link_idx, 3:7]
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32)
    axis_w = quat_apply(link_quat_w, normal_l.unsqueeze(0).expand(env.num_envs, -1))
    axis_w = axis_w / torch.clamp(torch.linalg.norm(axis_w, dim=-1, keepdim=True), min=1.0e-8)
    env._rl_fixed_presser_key_base = key_base
    env._rl_fixed_presser_reset_step = global_step
    env._rl_dynamic_axis_axis_w = axis_w.detach().clone()
    env._rl_dynamic_axis_disp_m = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)


def ensure_dynamic_axis_presser(
    env,
    link_name: str,
    center_l_np: np.ndarray,
    normal_l_np: np.ndarray,
    quat_l_np: np.ndarray,
    *,
    offset_m: float,
    key_base: tuple[str, str],
    force_reset: bool,
    drive_sign: float = 1.0,
    max_travel_m: float | None = None,
    slide_axis_l_np: np.ndarray | None = None,
    max_slide_m: float = 0.0,
) -> None:
    global_step = int(getattr(env, "_rl_tactile_viz_global_step", -1))
    reset_step = getattr(env, "_rl_dynamic_axis_reset_step", None)
    if getattr(env, "_rl_dynamic_axis_key_base", None) == key_base and not (force_reset and reset_step != global_step):
        return

    write_object_pose(env, link_name, center_l_np, normal_l_np, quat_l_np, offset_m=offset_m)
    obj: RigidObject = env.scene["object"]
    obj.update(0.0)

    robot: Articulation = env.scene["robot"]
    link_idx = body_index(robot, link_name)
    link_quat_w = robot.data.body_link_state_w[:, link_idx, 3:7]
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32)
    axis_w = quat_apply(link_quat_w, normal_l.unsqueeze(0).expand(env.num_envs, -1))
    axis_w = axis_w / torch.clamp(torch.linalg.norm(axis_w, dim=-1, keepdim=True), min=1.0e-8)
    slide_axis_w = None
    if slide_axis_l_np is not None and float(max_slide_m) > 0.0:
        slide_axis_l = torch.as_tensor(slide_axis_l_np, device=env.device, dtype=torch.float32)
        slide_axis_w = quat_apply(link_quat_w, slide_axis_l.unsqueeze(0).expand(env.num_envs, -1))
        slide_axis_w = slide_axis_w - axis_w * torch.sum(slide_axis_w * axis_w, dim=-1, keepdim=True)
        slide_axis_w = slide_axis_w / torch.clamp(
            torch.linalg.norm(slide_axis_w, dim=-1, keepdim=True), min=1.0e-8
        )

    env._rl_dynamic_axis_key_base = key_base
    env._rl_dynamic_axis_reset_step = global_step
    env._rl_dynamic_axis_anchor_w = obj.data.root_pos_w.detach().clone()
    env._rl_dynamic_axis_axis_w = axis_w.detach().clone()
    env._rl_dynamic_axis_drive_axis_w = (axis_w * float(drive_sign)).detach().clone()
    env._rl_dynamic_axis_slide_axis_w = None if slide_axis_w is None else slide_axis_w.detach().clone()
    env._rl_dynamic_axis_quat_w = obj.data.root_quat_w.detach().clone()
    env._rl_dynamic_axis_max_travel_m = (
        max(0.0, float(args_cli.presser_axis_max_travel))
        if max_travel_m is None
        else max(0.0, float(max_travel_m))
    )
    env._rl_dynamic_axis_disp_m = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    env._rl_dynamic_axis_max_slide_m = max(0.0, float(max_slide_m))
    env._rl_dynamic_axis_slide_disp_m = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    env._rl_dynamic_axis_target_slide_m = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


def drive_dynamic_axis_presser(env, target_disp_m: float | None, *, target_slide_m: float = 0.0) -> None:
    anchor_w = getattr(env, "_rl_dynamic_axis_anchor_w", None)
    drive_axis_w = getattr(env, "_rl_dynamic_axis_drive_axis_w", None)
    if not (isinstance(anchor_w, torch.Tensor) and isinstance(drive_axis_w, torch.Tensor)):
        return
    obj: RigidObject = env.scene["object"]
    obj.update(0.0)
    pos_w = obj.data.root_pos_w
    vel_w = obj.data.root_vel_w[:, :3]
    disp = torch.sum((pos_w - anchor_w) * drive_axis_w, dim=-1, keepdim=True)
    dt = max(float(getattr(env, "physics_dt", 1.0 / 120.0)), 1.0e-6)
    max_speed = max(0.0, float(args_cli.presser_axis_max_speed))
    if target_disp_m is None:
        speed = torch.sum(vel_w * drive_axis_w, dim=-1, keepdim=True)
    else:
        max_travel = float(getattr(env, "_rl_dynamic_axis_max_travel_m", 0.0))
        target = torch.full_like(disp, min(max(0.0, float(target_disp_m)), max_travel))
        speed = (target - disp) / dt
        if max_speed > 0.0:
            speed = torch.clamp(speed, min=-max_speed, max=max_speed)
    lin_vel_w = drive_axis_w * speed
    slide_axis_w = getattr(env, "_rl_dynamic_axis_slide_axis_w", None)
    if isinstance(slide_axis_w, torch.Tensor):
        slide_disp = torch.sum((pos_w - anchor_w) * slide_axis_w, dim=-1, keepdim=True)
        max_slide = float(getattr(env, "_rl_dynamic_axis_max_slide_m", 0.0))
        slide_target = torch.full_like(slide_disp, min(max(0.0, float(target_slide_m)), max_slide))
        env._rl_dynamic_axis_target_slide_m = slide_target.detach().clone()
        slide_speed = (slide_target - slide_disp) / dt
        if max_speed > 0.0:
            slide_speed = torch.clamp(slide_speed, min=-max_speed, max=max_speed)
        lin_vel_w = lin_vel_w + slide_axis_w * slide_speed
    root_vel_w = torch.cat((lin_vel_w, torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)), dim=-1)
    obj.write_root_velocity_to_sim(root_vel_w)
    obj.update(0.0)


def project_dynamic_axis_presser(env) -> None:
    if not dynamic_axis_enabled():
        return
    anchor_w = getattr(env, "_rl_dynamic_axis_anchor_w", None)
    drive_axis_w = getattr(env, "_rl_dynamic_axis_drive_axis_w", None)
    quat_w = getattr(env, "_rl_dynamic_axis_quat_w", None)
    if not (
        isinstance(anchor_w, torch.Tensor)
        and isinstance(drive_axis_w, torch.Tensor)
        and isinstance(quat_w, torch.Tensor)
    ):
        return

    obj: RigidObject = env.scene["object"]
    obj.update(0.0)
    pos_w = obj.data.root_pos_w
    vel_w = obj.data.root_vel_w
    disp = torch.sum((pos_w - anchor_w) * drive_axis_w, dim=-1, keepdim=True)
    max_travel = max(0.0, float(getattr(env, "_rl_dynamic_axis_max_travel_m", 0.0)))
    disp = torch.clamp(disp, min=0.0, max=max_travel)
    locked_pos_w = anchor_w + drive_axis_w * disp
    vel_axis = torch.sum(vel_w[:, :3] * drive_axis_w, dim=-1, keepdim=True)
    at_min = disp <= 1.0e-8
    at_max = disp >= max_travel - 1.0e-8
    vel_axis = torch.where((at_min & (vel_axis < 0.0)) | (at_max & (vel_axis > 0.0)), torch.zeros_like(vel_axis), vel_axis)
    lin_vel_w = drive_axis_w * vel_axis
    slide_axis_w = getattr(env, "_rl_dynamic_axis_slide_axis_w", None)
    slide_disp = torch.zeros_like(disp)
    if isinstance(slide_axis_w, torch.Tensor):
        max_slide = max(0.0, float(getattr(env, "_rl_dynamic_axis_max_slide_m", 0.0)))
        slide_disp = getattr(env, "_rl_dynamic_axis_target_slide_m", None)
        if not isinstance(slide_disp, torch.Tensor):
            slide_disp = torch.sum((pos_w - anchor_w) * slide_axis_w, dim=-1, keepdim=True)
        slide_disp = torch.clamp(slide_disp, min=0.0, max=max_slide)
        locked_pos_w = locked_pos_w + slide_axis_w * slide_disp
        vel_slide = torch.sum(vel_w[:, :3] * slide_axis_w, dim=-1, keepdim=True)
        at_slide_min = slide_disp <= 1.0e-8
        at_slide_max = slide_disp >= max_slide - 1.0e-8
        vel_slide = torch.where(
            (at_slide_min & (vel_slide < 0.0)) | (at_slide_max & (vel_slide > 0.0)),
            torch.zeros_like(vel_slide),
            vel_slide,
        )
        lin_vel_w = lin_vel_w + slide_axis_w * vel_slide
    root_vel_w = torch.cat((lin_vel_w, torch.zeros_like(vel_w[:, 3:])), dim=-1)
    obj.write_root_pose_to_sim(torch.cat((locked_pos_w, quat_w), dim=-1))
    obj.write_root_velocity_to_sim(root_vel_w)
    obj.update(0.0)
    env._rl_dynamic_axis_disp_m = disp.squeeze(-1).detach()
    env._rl_dynamic_axis_slide_disp_m = slide_disp.squeeze(-1).detach()


def apply_dynamic_axis_presser_force(env) -> None:
    if not presser_force_enabled():
        return
    drive_axis_w = getattr(env, "_rl_dynamic_axis_drive_axis_w", None)
    if not isinstance(drive_axis_w, torch.Tensor):
        return
    obj: RigidObject = env.scene["object"]
    force_n = max(0.0, float(args_cli.presser_force_n))
    ramp_steps = int(args_cli.presser_force_ramp_steps)
    if ramp_steps > 0:
        global_step = max(0, int(getattr(env, "_rl_tactile_viz_global_step", 0)))
        force_n *= min(1.0, global_step / float(ramp_steps))
    num_bodies = max(1, int(getattr(obj, "num_bodies", 1)))
    forces_w = drive_axis_w[:, None, :].expand(env.num_envs, num_bodies, 3) * force_n
    torques_w = torch.zeros_like(forces_w)
    composer = getattr(obj, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "set_forces_and_torques"):
        if hasattr(composer, "reset"):
            composer.reset()
        composer.set_forces_and_torques(forces=forces_w, torques=torques_w, is_global=True)
    elif hasattr(obj, "set_external_force_and_torque"):
        obj.set_external_force_and_torque(forces=forces_w, torques=torques_w, is_global=True)
    env._rl_dynamic_axis_force_n = force_n


def update_focus_contact_stats(env, link_name: str) -> None:
    sensor = env.scene.sensors.get(f"{link_name}_object_s")
    force_matrix = None if sensor is None else getattr(getattr(sensor, "data", None), "force_matrix_w", None)
    if not isinstance(force_matrix, torch.Tensor) or force_matrix.numel() == 0:
        env._rl_focus_contact_force_n = 0.0
        env._rl_focus_normal_force_n = 0.0
        return
    force_w = torch.nan_to_num(force_matrix.reshape(env.num_envs, -1, 3)).sum(dim=1)
    axis_w = getattr(env, "_rl_dynamic_axis_axis_w", None)
    if isinstance(axis_w, torch.Tensor) and axis_w.shape == force_w.shape:
        normal_force = torch.abs(torch.sum(force_w * axis_w, dim=-1))
    else:
        normal_force = torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)
    contact_force_n, normal_force_n = torch.stack(
        (torch.linalg.norm(force_w[0]), normal_force[0])
    ).detach().cpu().tolist()
    env._rl_focus_contact_force_n = float(contact_force_n)
    env._rl_focus_normal_force_n = float(normal_force_n)


def sync_pressure_sensors_to_object(env) -> None:
    obj: RigidObject = env.scene["object"]
    for name in TIANJI_PRESSURE_SENSOR_NAMES:
        sensor = env.scene.sensors.get(str(name))
        if sensor is None:
            continue
        if getattr(sensor, "_target_mesh_prim_path", None) is not None and hasattr(sensor, "set_target_pose"):
            sensor.set_target_pose(obj.data.root_pos_w, obj.data.root_quat_w)
        elif hasattr(sensor, "set_box_pose"):
            sensor.set_box_pose(obj.data.root_pos_w, obj.data.root_quat_w)
        outdated = getattr(sensor, "_is_outdated", None)
        if isinstance(outdated, torch.Tensor):
            outdated[:] = True
        if hasattr(sensor, "update"):
            sensor.update(0.0, force_recompute=True)


def pressure_maps_from_env(env) -> list[torch.Tensor]:
    # Prefer the exact post-Gaussian tensor emitted by rl_ours_pressure.
    # The visualizer explicitly enables this detached cache; normal training
    # does not create it. Fall back to sensor caches only for non-ours tasks.
    maps: list[torch.Tensor] = []
    sensor_counts: list[int] = []
    sensor_sources: list[torch.Tensor | None] = []
    for name in TIANJI_PRESSURE_SENSOR_NAMES:
        sensor = env.scene.sensors.get(str(name))
        data = None if sensor is None else getattr(sensor, "data", None)
        force_map = None if data is None else getattr(data, "pressure_force_map", None)
        if force_map is None:
            sensor_counts.append(0)
            sensor_sources.append(None)
            continue
        source = force_map.detach()
        if source.ndim == 4:
            source = source[:, 0]
        source = source[: env.num_envs].reshape(min(env.num_envs, int(source.shape[0])), -1)
        sensor_counts.append(int(source.shape[-1]))
        sensor_sources.append(source)

    policy_pressure = getattr(env, "_rl_pressure_observation", None)
    expected = sum(sensor_counts)
    if (
        isinstance(policy_pressure, torch.Tensor)
        and policy_pressure.ndim == 2
        and int(policy_pressure.shape[0]) >= env.num_envs
        and int(policy_pressure.shape[1]) == expected
        and all(count > 0 for count in sensor_counts)
    ):
        policy_pressure = torch.nan_to_num(
            policy_pressure[: env.num_envs].detach().to(device=env.device, dtype=torch.float32)
        )
        env._rl_pressure_visual_source = "policy_observation"
        return list(torch.split(policy_pressure, sensor_counts, dim=1))

    env._rl_pressure_visual_source = "sensor_cache_fallback"
    for source, count in zip(sensor_sources, sensor_counts, strict=True):
        if source is None or count <= 0:
            maps.append(torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32))
            continue
        maps.append(torch.nan_to_num(source.to(device=env.device, dtype=torch.float32)))
    return maps


def build_pressure_display_diffusion_kernel(
    points_l: np.ndarray,
    normals_l: np.ndarray,
    *,
    sigma_m: float,
    radius_sigma: float = 3.0,
    normal_power: float = 1.0,
) -> np.ndarray:
    """Build a CPU-only, force-conserving display kernel from physical taxel geometry."""

    points = np.asarray(points_l, dtype=np.float32)
    normals = np.asarray(normals_l, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"pressure taxel points must be finite (P, 3), got {points.shape}")
    if normals.shape != points.shape or not np.isfinite(normals).all():
        raise ValueError(f"pressure taxel normals must match finite points, got {normals.shape}/{points.shape}")

    sigma = float(sigma_m)
    radius_scale = float(radius_sigma)
    alignment_power = float(normal_power)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("display diffusion sigma_m must be finite and positive")
    if not np.isfinite(radius_scale) or radius_scale <= 0.0:
        raise ValueError("display diffusion radius_sigma must be finite and positive")
    if not np.isfinite(alignment_power) or alignment_power < 0.0:
        raise ValueError("display diffusion normal_power must be finite and non-negative")

    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(normal_norm <= 1.0e-8):
        raise ValueError("pressure taxel normals must be non-zero")
    normals = normals / normal_norm

    # K[i, j] is the fraction of source taxel j displayed at target taxel i.
    delta = points[:, None, :] - points[None, :, :]
    normal_i = normals[:, None, :]
    normal_j = normals[None, :, :]
    tangent_i = delta - np.sum(delta * normal_i, axis=-1, keepdims=True) * normal_i
    tangent_j = delta - np.sum(delta * normal_j, axis=-1, keepdims=True) * normal_j
    distance_sq = 0.5 * (
        np.sum(tangent_i * tangent_i, axis=-1) + np.sum(tangent_j * tangent_j, axis=-1)
    )

    alignment = np.clip(normals @ normals.T, 0.0, 1.0) ** alignment_power
    weights = np.exp(-0.5 * distance_sq / max(sigma * sigma, 1.0e-12)) * alignment
    radius_m = radius_scale * sigma
    weights = np.where(distance_sq <= radius_m * radius_m, weights, 0.0)
    np.fill_diagonal(weights, 1.0)

    # Normalize each source column so diffusion does not inflate the displayed total.
    denominator = np.sum(weights, axis=0, keepdims=True)
    kernel = weights / np.maximum(denominator, 1.0e-8)
    return np.asarray(kernel, dtype=np.float32)


def diffuse_pressure_display_values(
    values: np.ndarray,
    kernel: np.ndarray,
    *,
    blend: float,
) -> np.ndarray:
    """Diffuse a CPU display copy without mutating the sensor or RL observation tensor."""

    raw = np.nan_to_num(
        np.asarray(values, dtype=np.float32).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).copy()
    matrix = np.asarray(kernel, dtype=np.float32)
    if matrix.shape != (raw.size, raw.size):
        raise ValueError(f"display diffusion kernel {matrix.shape} does not match {raw.size} pressure taxels")
    amount = float(blend)
    if not np.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("display diffusion blend must be finite and within [0, 1]")
    if amount <= 0.0:
        return np.maximum(raw, 0.0)

    spread = matrix @ raw
    display = (1.0 - amount) * raw + amount * spread
    return np.maximum(np.asarray(display, dtype=np.float32), 0.0)


def pressure_layout_pixel_centers(points_l: np.ndarray, *, spacing_px: int) -> tuple[np.ndarray, int, int]:
    """Project link-local taxels to image pixels: +Y right, +Z up."""

    points = np.asarray(points_l, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"pressure taxel points must be finite (P, 3), got {points.shape}")
    yz = points[:, 1:3]
    distances = np.linalg.norm(yz[:, None] - yz[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1.0e-9)]
    physical_spacing = float(np.median(positive)) if positive.size else 1.0
    spacing = max(2, int(spacing_px))
    pixels_per_unit = float(spacing) / physical_spacing
    padding = max(1, spacing // 2)
    rows = np.rint((float(np.max(yz[:, 1])) - yz[:, 1]) * pixels_per_unit + padding).astype(np.int64)
    cols = np.rint((yz[:, 0] - float(np.min(yz[:, 0]))) * pixels_per_unit + padding).astype(np.int64)
    centers = np.column_stack((rows, cols))
    return centers, int(np.max(rows) + padding + 1), int(np.max(cols) + padding + 1)


def pressure_layout_pane(
    force_map: torch.Tensor | np.ndarray,
    points_l: np.ndarray,
    *,
    diffusion_kernel: np.ndarray | None = None,
    diffusion_blend: float = 0.0,
    spacing_px: int,
    gamma: float,
) -> np.ndarray:
    flat = force_map.reshape(-1)
    centers, height, width = pressure_layout_pixel_centers(points_l, spacing_px=spacing_px)
    flat_count = int(flat.numel()) if isinstance(flat, torch.Tensor) else int(np.asarray(flat).size)
    if flat_count != centers.shape[0]:
        raise ValueError(f"pressure map has {flat_count} values for {centers.shape[0]} taxel positions")
    flat_values = flat.detach().cpu().numpy() if isinstance(flat, torch.Tensor) else np.asarray(flat)
    display_values = np.nan_to_num(
        np.asarray(flat_values, dtype=np.float32).copy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if diffusion_kernel is not None:
        display_values = diffuse_pressure_display_values(
            display_values,
            diffusion_kernel,
            blend=diffusion_blend,
        )
    colors = jet_colormap(normalize_grid(torch.from_numpy(display_values), gamma)).numpy()
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    radius = max(1, int(spacing_px) // 4)
    for (row, col), color in zip(centers, colors, strict=True):
        image[
            max(0, int(row) - radius) : min(height, int(row) + radius + 1),
            max(0, int(col) - radius) : min(width, int(col) + radius + 1),
        ] = color
    return image


def tacmap_maps_from_env(env) -> torch.Tensor:
    maps = getattr(env, "_rl_tacmap_penetration_m", None)
    if isinstance(maps, torch.Tensor):
        return torch.nan_to_num(maps.detach().to(device=env.device, dtype=torch.float32))
    return torch.zeros((env.num_envs, len(FINGER_CHOICES), TIANJI_TACMAP_RL_ROWS, TIANJI_TACMAP_RL_COLS), device=env.device)


def tacsl_baseline_is_enabled(env) -> bool:
    proprio_cfg = getattr(getattr(env, "cfg", None), "observations", None)
    proprio_cfg = getattr(proprio_cfg, "proprio", None)
    return getattr(proprio_cfg, "rl_tacsl_baseline", None) is not None


def hydroshear_baseline_is_enabled(env) -> bool:
    proprio_cfg = getattr(getattr(env, "cfg", None), "observations", None)
    proprio_cfg = getattr(proprio_cfg, "proprio", None)
    return getattr(proprio_cfg, "rl_hydroshear_baseline", None) is not None


def fots_baseline_is_enabled(env) -> bool:
    proprio_cfg = getattr(getattr(env, "cfg", None), "observations", None)
    proprio_cfg = getattr(proprio_cfg, "proprio", None)
    return getattr(proprio_cfg, "rl_fots_baseline", None) is not None


def tacsl_baseline_grid_from_env(env) -> torch.Tensor | None:
    values = getattr(env, "_brainco_rl_tacsl_baseline_obs", None)
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[0] < 1:
        return None
    expected = (
        len(FINGER_CHOICES)
        * int(TIANJI_TACSL_BASELINE_OUTPUT_ROWS)
        * int(TIANJI_TACSL_BASELINE_OUTPUT_COLS)
        * 3
    )
    if int(values.shape[1]) != expected:
        return None
    return torch.nan_to_num(values.detach()).reshape(
        values.shape[0],
        len(FINGER_CHOICES),
        int(TIANJI_TACSL_BASELINE_OUTPUT_ROWS),
        int(TIANJI_TACSL_BASELINE_OUTPUT_COLS),
        3,
    )


def tacsl_baseline_renderer(env) -> RevoTacslShearAdapter:
    renderer = getattr(env, "_rl_tacsl_baseline_renderer", None)
    if renderer is None:
        renderer = RevoTacslShearAdapter(
            RevoTacslShearCfg(
                width=int(TIANJI_TACSL_BASELINE_RAY_COLS),
                height=int(TIANJI_TACSL_BASELINE_RAY_ROWS),
                render_rows=int(TIANJI_TACSL_BASELINE_OUTPUT_ROWS),
                render_cols=int(TIANJI_TACSL_BASELINE_OUTPUT_COLS),
                device=str(env.device),
            )
        )
        env._rl_tacsl_baseline_renderer = renderer
    return renderer


def build_tacsl_baseline_image(env) -> tuple[np.ndarray, str]:
    grid = tacsl_baseline_grid_from_env(env)
    if grid is None:
        blank = np.zeros(
            (int(TIANJI_TACSL_BASELINE_RAY_ROWS), int(TIANJI_TACSL_BASELINE_RAY_COLS), 3),
            dtype=np.uint8,
        )
        return tile_rgb([blank] * len(FINGER_CHOICES), cols=5), "tacsl=waiting"

    field = grid[0]
    rendered = tacsl_baseline_renderer(env).step(field[..., 0], field[..., 1:3])
    panes = [to_numpy_uint8_rgb(image) for image in rendered.shear_images]
    if not bool(args_cli.show_all_tacmap_fingers):
        focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
        panes = [pane if index == focus_idx else np.zeros_like(pane) for index, pane in enumerate(panes)]
    image = tile_rgb(panes, cols=5)

    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    focus = field[focus_idx]
    normal = focus[..., 0]
    shear = torch.linalg.norm(focus[..., 1:3], dim=-1)
    active = (normal > 0.0) | (shear > 0.0)
    stats = (
        f"target={getattr(env, '_rl_tactile_viz_phase', 'unknown')} "
        f"implementation=tacsl_baseline focus={args_cli.focus_finger} "
        f"active={int(torch.count_nonzero(active).detach().cpu())}/"
        f"{int(TIANJI_TACSL_BASELINE_OUTPUT_ROWS) * int(TIANJI_TACSL_BASELINE_OUTPUT_COLS)} "
        f"normal_max={float(torch.amax(normal).detach().cpu()):.6f} "
        f"shear_max={float(torch.amax(shear).detach().cpu()):.6f} "
        f"ray_field=5x{int(TIANJI_TACSL_BASELINE_RAY_ROWS)}x{int(TIANJI_TACSL_BASELINE_RAY_COLS)} "
        f"policy_obs=5x{int(TIANJI_TACSL_BASELINE_OUTPUT_ROWS)}x"
        f"{int(TIANJI_TACSL_BASELINE_OUTPUT_COLS)}x3"
    )
    return image, stats


def build_hydroshear_baseline_image(env) -> tuple[np.ndarray, str]:
    native_blank = np.zeros(
        (int(TIANJI_HYDROSHEAR_BASELINE_RENDER_ROWS), int(TIANJI_HYDROSHEAR_BASELINE_RENDER_COLS), 3),
        dtype=np.uint8,
    )
    blank = hydroshear_marker_pane(native_blank)
    output = getattr(env, "_rl_hydroshear_baseline_output", None)
    if output is None or not bool(args_cli.show_hydroshear_marker):
        return blank, "implementation=hydroshear_baseline marker=waiting"

    images = getattr(output, "marker_images", None)
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    focus_image = image_from_sensor_batch(images, focus_idx)
    image = blank if focus_image is None else hydroshear_marker_pane(focus_image)

    displacement = getattr(env, "_rl_hydroshear_baseline_displacement_m", None)
    active = 0
    max_displacement_mm = 0.0
    if isinstance(displacement, torch.Tensor) and displacement.ndim == 4 and displacement.shape[0] > 0:
        focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
        focus = torch.nan_to_num(displacement[0, focus_idx])
        magnitude = torch.linalg.norm(focus, dim=-1)
        active = int(torch.count_nonzero(magnitude > 0.0).detach().cpu())
        max_displacement_mm = float(torch.amax(magnitude).detach().cpu() * 1000.0)

    adapter = getattr(env, "_brainco_rl_hydroshear_baseline_adapter", None)
    samples = getattr(adapter, "_object_sample_points_l", None)
    sample_count = int(samples.shape[0]) if isinstance(samples, torch.Tensor) else 0
    stats = (
        f"target={getattr(env, '_rl_tactile_viz_phase', 'unknown')} "
        f"implementation=hydroshear_baseline focus={args_cli.focus_finger} "
        f"active={active}/{int(TIANJI_HYDROSHEAR_BASELINE_MARKER_ROWS) * int(TIANJI_HYDROSHEAR_BASELINE_MARKER_COLS)} "
        f"marker_max={max_displacement_mm:.6f}mm samples={sample_count} "
        f"ray_field=5x{int(TIANJI_HYDROSHEAR_BASELINE_RAY_ROWS)}x"
        f"{int(TIANJI_HYDROSHEAR_BASELINE_RAY_COLS)} "
        f"policy_obs=5x{int(TIANJI_HYDROSHEAR_BASELINE_MARKER_ROWS)}x"
        f"{int(TIANJI_HYDROSHEAR_BASELINE_MARKER_COLS)}x3"
    )
    return image, stats


def fots_baseline_renderer(env) -> RevoFotsAdapter:
    renderer = getattr(env, "_rl_fots_baseline_renderer", None)
    if renderer is None:
        renderer = RevoFotsAdapter(
            RevoFotsCfg(
                width=int(TIANJI_FOTS_BASELINE_RENDER_COLS),
                height=int(TIANJI_FOTS_BASELINE_RENDER_ROWS),
                marker_rows=int(TIANJI_FOTS_BASELINE_MARKER_ROWS),
                marker_cols=int(TIANJI_FOTS_BASELINE_MARKER_COLS),
                marker_margin_x=float(TIANJI_FOTS_BASELINE_MARKER_MARGIN_X),
                marker_margin_y=float(TIANJI_FOTS_BASELINE_MARKER_MARGIN_Y),
                mm2pix=float(TIANJI_FOTS_BASELINE_MM2PIX),
                lamb=tuple(float(value) for value in TIANJI_FOTS_BASELINE_LAMB),
                contact_threshold_mm=float(TIANJI_FOTS_BASELINE_CONTACT_THRESHOLD_MM),
                track_contact_center=True,
                render_marker_image=False,
                render_overlay_image=False,
                device=str(env.device),
            )
        )
        env._rl_fots_baseline_renderer = renderer
    return renderer


def build_fots_baseline_image(env) -> tuple[np.ndarray, str]:
    blank = np.zeros(
        (int(TIANJI_FOTS_BASELINE_RENDER_ROWS), int(TIANJI_FOTS_BASELINE_RENDER_COLS), 3),
        dtype=np.uint8,
    )
    output = getattr(env, "_rl_fots_baseline_output", None)
    flow = None if output is None else getattr(output, "marker_flow", None)
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    if not isinstance(flow, torch.Tensor) or flow.ndim != 4 or flow.shape[0] <= focus_idx:
        return blank, "implementation=fots_baseline marker=waiting"

    focus_flow = flow[focus_idx]
    image = fots_baseline_renderer(env).render_markers(focus_flow)
    displacement = torch.nan_to_num(focus_flow[1] - focus_flow[0])
    magnitude = torch.linalg.norm(displacement, dim=-1)
    active = int(torch.count_nonzero(magnitude > 0.0).detach().cpu())
    max_motion_px = float(torch.amax(magnitude).detach().cpu()) if magnitude.numel() else 0.0
    max_depth = getattr(output, "max_depth_mm", None)
    max_depth_mm = (
        float(max_depth[focus_idx].detach().cpu())
        if isinstance(max_depth, torch.Tensor) and max_depth.numel() > focus_idx
        else 0.0
    )
    stats = (
        f"target={getattr(env, '_rl_tactile_viz_phase', 'unknown')} "
        f"implementation=fots_baseline focus={args_cli.focus_finger} "
        f"active={active}/{int(TIANJI_FOTS_BASELINE_MARKER_ROWS) * int(TIANJI_FOTS_BASELINE_MARKER_COLS)} "
        f"marker_max={max_motion_px:.6f}px depth_max={max_depth_mm:.6f}mm "
        f"ray_field=5x{int(TIANJI_FOTS_BASELINE_RAY_ROWS)}x{int(TIANJI_FOTS_BASELINE_RAY_COLS)} "
        f"policy_obs=5x{int(TIANJI_FOTS_BASELINE_MARKER_ROWS)}x"
        f"{int(TIANJI_FOTS_BASELINE_MARKER_COLS)}x2"
    )
    return image, stats


def focus_tacmap_is_active(env) -> bool:
    tacsl_grid = tacsl_baseline_grid_from_env(env)
    if tacsl_grid is not None:
        idx = FINGER_CHOICES.index(args_cli.focus_finger)
        return bool(torch.any(tacsl_grid[0, idx, ..., 0] > 0.0).detach().cpu())
    maps = tacmap_maps_from_env(env)
    idx = FINGER_CHOICES.index(args_cli.focus_finger)
    if maps.ndim != 4 or maps.shape[1] <= idx:
        return False
    threshold_m = max(0.0, float(args_cli.tacmap_onset_threshold_mm)) * 1.0e-3
    return bool(torch.any(maps[0, idx] > threshold_m).detach().cpu())


def display_tacmap_maps(maps: torch.Tensor) -> torch.Tensor:
    out = maps
    rot90 = int(args_cli.tacmap_display_rot90) % 4
    if rot90:
        out = torch.rot90(out, k=rot90, dims=(-2, -1))
    if bool(args_cli.tacmap_display_flip_y):
        out = torch.flip(out, dims=(-2,))
    if bool(args_cli.tacmap_display_flip_x):
        out = torch.flip(out, dims=(-1,))
    return out


def display_tacmap_vec_maps(maps: torch.Tensor) -> torch.Tensor:
    out = maps
    rot90 = int(args_cli.tacmap_display_rot90) % 4
    if rot90:
        out = torch.rot90(out, k=rot90, dims=(-3, -2))
    if bool(args_cli.tacmap_display_flip_y):
        out = torch.flip(out, dims=(-3,))
    if bool(args_cli.tacmap_display_flip_x):
        out = torch.flip(out, dims=(-2,))
    return out


def calibrated_marker_tacmap_grid_layout() -> tuple[np.ndarray, np.ndarray]:
    """Return Marker positions as continuous ``(row, col)`` coordinates in the 32x24 TacMap grid."""

    cached = getattr(calibrated_marker_tacmap_grid_layout, "_cache", None)
    if cached is not None:
        return cached

    from scripts.project_vitai_markers_to_mesh import camera_xy_to_ray_grid_coordinates

    coordinate_sets: list[np.ndarray] = []
    valid_sets: list[np.ndarray] = []
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        for finger in FINGER_CHOICES:
            hull_xy = np.asarray(layout[f"{finger}_camera_visible_hull_xy_m"], dtype=np.float32).reshape(-1, 2)
            marker_xy = np.asarray(layout[f"{finger}_points_camera_m"], dtype=np.float32).reshape(-1, 3)[:, :2]
            coordinates, inside = camera_xy_to_ray_grid_coordinates(
                hull_xy,
                marker_xy,
                rows=TIANJI_TACMAP_RL_ROWS,
                cols=TIANJI_TACMAP_RL_COLS,
            )
            marker_valid = np.asarray(layout[f"{finger}_marker_ray_valid"], dtype=bool).reshape(-1)
            if not (len(coordinates) == len(inside) == len(marker_valid)):
                raise ValueError(f"Marker/TacMap layout length mismatch for {finger}")
            coordinate_sets.append(coordinates)
            valid_sets.append(inside & marker_valid)
    cached = (
        np.stack(coordinate_sets, axis=0).astype(np.float32),
        np.stack(valid_sets, axis=0),
    )
    calibrated_marker_tacmap_grid_layout._cache = cached
    return cached


def display_tacmap_point_coords(
    coords: torch.Tensor,
    *,
    source_rows: int,
    source_cols: int,
) -> torch.Tensor:
    """Apply the configured TacMap display rotation/flips to continuous point coordinates."""

    out = coords.to(dtype=torch.float32).clone()
    rows = int(source_rows)
    cols = int(source_cols)
    rot90 = int(args_cli.tacmap_display_rot90) % 4
    if rot90 == 1:
        row = cols - 1.0 - out[..., 1]
        col = out[..., 0]
        out = torch.stack((row, col), dim=-1)
        rows, cols = cols, rows
    elif rot90 == 2:
        out = torch.stack((rows - 1.0 - out[..., 0], cols - 1.0 - out[..., 1]), dim=-1)
    elif rot90 == 3:
        row = out[..., 1]
        col = rows - 1.0 - out[..., 0]
        out = torch.stack((row, col), dim=-1)
        rows, cols = cols, rows
    if bool(args_cli.tacmap_display_flip_y):
        out[..., 0] = rows - 1.0 - out[..., 0]
    if bool(args_cli.tacmap_display_flip_x):
        out[..., 1] = cols - 1.0 - out[..., 1]
    return out


def visual_local_tacmap_enabled() -> bool:
    """Return whether this visualizer should create display-only local TacMap rays."""

    return bool(args_cli.tacmap_local_refinement) and str(args_cli.target) in ("tacmap", "cycle")


def connected_contact_bbox(
    depth_m: np.ndarray,
    *,
    threshold_m: float,
) -> tuple[int, int, int, int] | None:
    """Return the 4-connected component bbox containing the maximum-depth cell."""

    depth = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if depth.ndim != 2 or depth.size == 0:
        return None
    contact = depth > max(0.0, float(threshold_m))
    if not np.any(contact):
        return None
    seed_flat = int(np.argmax(np.where(contact, depth, -np.inf)))
    seed_row, seed_col = np.unravel_index(seed_flat, depth.shape)
    visited = np.zeros_like(contact)
    visited[seed_row, seed_col] = True
    stack = [(int(seed_row), int(seed_col))]
    while stack:
        row, col = stack.pop()
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (
                0 <= next_row < depth.shape[0]
                and 0 <= next_col < depth.shape[1]
                and contact[next_row, next_col]
                and not visited[next_row, next_col]
            ):
                visited[next_row, next_col] = True
                stack.append((next_row, next_col))
    rows, cols = np.nonzero(visited)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def visual_local_tacmap_contact_roi(
    env,
    state: dict,
) -> tuple[float, float, float, float] | None:
    """Map the native 32x24 contact box to an equally expanded camera-XY metric ROI."""

    focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
    base = torch.nan_to_num(tacmap_maps_from_env(env)[0, focus_index].detach()).clone()
    bbox = connected_contact_bbox(
        base.detach().cpu().numpy(),
        threshold_m=float(args_cli.tacmap_local_contact_threshold_mm) * 1.0e-3,
    )
    if bbox is None:
        return None
    row_min, row_max, col_min, col_max = bbox

    native_row = np.asarray(state["reference_native_row"], dtype=np.float32).reshape(-1)
    native_col = np.asarray(state["reference_native_col"], dtype=np.float32).reshape(-1)
    reference_xy = np.asarray(state["reference_xy_camera_np"], dtype=np.float32)
    if reference_xy.shape != (len(native_row), len(native_col), 2):
        raise ValueError("Dense-reference camera XY does not match its native-coordinate topology")
    row_inside = (native_row >= float(row_min) - 0.5) & (native_row <= float(row_max) + 0.5)
    col_inside = (native_col >= float(col_min) - 0.5) & (native_col <= float(col_max) + 0.5)
    seed_xy = reference_xy[row_inside[:, None] & col_inside[None, :]]
    if len(seed_xy) == 0:
        return None

    margin_m = max(0.0, float(args_cli.tacmap_local_roi_margin_mm)) * 1.0e-3
    full_x_min, full_x_max, full_y_min, full_y_max = (
        float(value) for value in state["full_camera_bounds_m"]
    )
    return (
        max(full_x_min, float(np.min(seed_xy[:, 0])) - margin_m),
        min(full_x_max, float(np.max(seed_xy[:, 0])) + margin_m),
        max(full_y_min, float(np.min(seed_xy[:, 1])) - margin_m),
        min(full_y_max, float(np.max(seed_xy[:, 1])) + margin_m),
    )


def visual_local_tacmap_integer_grid_shape(
    pixel_rows: int,
    pixel_cols: int,
    sample_count: int,
) -> tuple[int, int]:
    """Choose a near-isotropic integer-pixel grid containing at least ``sample_count`` points."""

    pixel_rows = max(1, int(pixel_rows))
    pixel_cols = max(1, int(pixel_cols))
    sample_count = max(1, min(int(sample_count), pixel_rows * pixel_cols))
    row_min = max(1, math.ceil(sample_count / pixel_cols))
    row_max = min(pixel_rows, sample_count)
    best_shape: tuple[int, int] | None = None
    best_score = math.inf
    for grid_rows in range(row_min, row_max + 1):
        grid_cols = math.ceil(sample_count / grid_rows)
        if grid_cols > pixel_cols:
            continue
        row_spacing = pixel_rows / float(grid_rows)
        col_spacing = pixel_cols / float(grid_cols)
        excess_ratio = (grid_rows * grid_cols - sample_count) / float(sample_count)
        score = abs(math.log(row_spacing / col_spacing)) + 0.01 * excess_ratio
        if score < best_score:
            best_score = score
            best_shape = (grid_rows, grid_cols)
    if best_shape is None:
        raise RuntimeError("Could not fit the local TacMap sample grid inside its integer-pixel ROI")
    return best_shape


def visual_local_tacmap_metric_grid_shape(
    width_m: float,
    height_m: float,
    sample_count: int,
    *,
    max_rows: int,
    max_cols: int,
) -> tuple[int, int]:
    """Choose a complete metric grid that never exceeds ``sample_count``."""

    width_m = max(float(width_m), 1.0e-9)
    height_m = max(float(height_m), 1.0e-9)
    max_rows = max(1, int(max_rows))
    max_cols = max(1, int(max_cols))
    sample_count = max(1, min(int(sample_count), max_rows * max_cols))
    row_max = min(max_rows, sample_count)
    best_shape: tuple[int, int] | None = None
    best_score = math.inf
    for grid_rows in range(1, row_max + 1):
        grid_cols = min(max_cols, sample_count // grid_rows)
        if grid_cols <= 0:
            continue
        y_spacing = height_m / float(grid_rows)
        x_spacing = width_m / float(grid_cols)
        unused_ratio = (sample_count - grid_rows * grid_cols) / float(sample_count)
        score = abs(math.log(y_spacing / x_spacing)) + unused_ratio
        if score < best_score:
            best_score = score
            best_shape = (grid_rows, grid_cols)
    if best_shape is None:
        raise RuntimeError("Could not fit the local TacMap metric grid inside its reference ROI")
    return best_shape


def sample_visual_local_reference(
    state: dict,
    bounds: tuple[float, float, float, float],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    tuple[int, int],
]:
    """Select exact integer pixels from the dense yellow-plane reference."""

    reference_depth = state["reference_depth_m"]
    reference_valid = state["reference_valid"]
    reference_starts = state["reference_starts_l"]
    if reference_starts.ndim != 3 or reference_starts.shape[-1] != 3:
        raise ValueError("Yellow-plane reference starts must have shape HxWx3")
    if reference_depth.shape != reference_starts.shape[:2] or reference_valid.shape != reference_depth.shape:
        raise ValueError("Yellow-plane reference depth/validity must match the HxW start grid")

    device = reference_depth.device
    reference_rows = int(reference_depth.shape[0])
    reference_cols = int(reference_depth.shape[1])
    reference_xy = state["reference_xy_camera_m"].to(device=device, dtype=torch.float32)
    if reference_xy.shape != (reference_rows, reference_cols, 2):
        raise ValueError("Dense-reference camera XY must have shape HxWx2")
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    roi_pixels = (
        (reference_xy[..., 0] >= x_min)
        & (reference_xy[..., 0] <= x_max)
        & (reference_xy[..., 1] >= y_min)
        & (reference_xy[..., 1] <= y_max)
    )
    roi_valid = roi_pixels & reference_valid.to(device=device, dtype=torch.bool)
    roi_indices = torch.nonzero(roi_valid.reshape(-1), as_tuple=False).reshape(-1)
    roi_count = int(roi_indices.numel())
    if roi_count <= 0:
        raise ValueError("Local TacMap ROI does not cover any yellow-plane pixels")

    ray_count = int(VISUAL_LOCAL_TACMAP_RAY_COUNT)
    if roi_count <= ray_count:
        selected_indices = roi_indices
        write_count = roi_count
        padding_count = ray_count - roi_count
        if padding_count > 0:
            padding = roi_indices.index_select(
                0,
                torch.arange(padding_count, device=device, dtype=torch.long) % roi_count,
            )
            selected_indices = torch.cat((selected_indices, padding), dim=0)
        grid_shape = (
            int(roi_valid.any(dim=1).count_nonzero()),
            int(torch.max(roi_valid.count_nonzero(dim=1)).item()),
        )
    else:
        roi_rows = torch.nonzero(roi_valid.any(dim=1), as_tuple=False).reshape(-1)
        max_cols = int(torch.max(roi_valid.count_nonzero(dim=1)).item())
        grid_rows, grid_cols = visual_local_tacmap_metric_grid_shape(
            x_max - x_min,
            y_max - y_min,
            ray_count,
            max_rows=int(roi_rows.numel()),
            max_cols=max_cols,
        )
        row_positions = torch.round(
            torch.linspace(0.0, float(roi_rows.numel() - 1), grid_rows, device=device)
        ).to(dtype=torch.long)
        selected_rows = roi_rows.index_select(0, row_positions)
        selected_row_valid = roi_valid.index_select(0, selected_rows)
        first_col = torch.argmax(selected_row_valid.to(dtype=torch.float32), dim=1)
        last_col = reference_cols - 1 - torch.argmax(
            torch.flip(selected_row_valid, dims=(1,)).to(dtype=torch.float32),
            dim=1,
        )
        col_fraction = torch.linspace(0.0, 1.0, grid_cols, device=device).reshape(1, -1)
        selected_cols = torch.round(
            first_col.to(dtype=torch.float32).unsqueeze(1)
            + col_fraction
            * (last_col - first_col).to(dtype=torch.float32).unsqueeze(1)
        ).to(dtype=torch.long)
        grid_indices = (selected_rows.unsqueeze(1) * reference_cols + selected_cols).reshape(-1)
        flat_roi_valid = roi_valid.reshape(-1)
        grid_indices = torch.unique(grid_indices[flat_roi_valid.index_select(0, grid_indices)])
        if int(grid_indices.numel()) > ray_count:
            selection_positions = torch.div(
                (2 * torch.arange(ray_count, device=device, dtype=torch.long) + 1)
                * int(grid_indices.numel()),
                2 * ray_count,
                rounding_mode="floor",
            )
            grid_indices = grid_indices.index_select(0, selection_positions)
        if int(grid_indices.numel()) < ray_count:
            selected_mask = torch.zeros(reference_rows * reference_cols, device=device, dtype=torch.bool)
            selected_mask[grid_indices] = True
            fill_count = ray_count - int(grid_indices.numel())
            interior = torch.zeros_like(roi_valid)
            if reference_rows > 2 and reference_cols > 2:
                interior[1:-1, 1:-1] = (
                    roi_valid[1:-1, 1:-1]
                    & roi_valid[:-2, 1:-1]
                    & roi_valid[2:, 1:-1]
                    & roi_valid[1:-1, :-2]
                    & roi_valid[1:-1, 2:]
                )
            interior_indices = torch.nonzero(interior.reshape(-1), as_tuple=False).reshape(-1)
            interior_indices = interior_indices[
                ~selected_mask.index_select(0, interior_indices)
            ]
            if int(interior_indices.numel()) >= fill_count:
                remaining = interior_indices
            else:
                remaining = roi_indices[~selected_mask.index_select(0, roi_indices)]
            fill_positions = torch.div(
                (2 * torch.arange(fill_count, device=device, dtype=torch.long) + 1)
                * int(remaining.numel()),
                2 * fill_count,
                rounding_mode="floor",
            )
            grid_indices = torch.cat((grid_indices, remaining.index_select(0, fill_positions)), dim=0)
        selected_indices = grid_indices
        write_count = ray_count
        grid_shape = (grid_rows, grid_cols)

    if int(selected_indices.numel()) != ray_count:
        raise RuntimeError(f"Integer-pixel TacMap selection produced {int(selected_indices.numel())} rays")
    starts = reference_starts.reshape(-1, 3).index_select(0, selected_indices)
    depth = reference_depth.reshape(-1).index_select(0, selected_indices)
    valid = reference_valid.reshape(-1).index_select(0, selected_indices)
    return starts, depth, valid, selected_indices, write_count, roi_valid, grid_shape


def rasterize_visual_local_tacmap_depth(
    state: dict,
    local_depth: torch.Tensor,
    local_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write local-ray depths directly back to their exact yellow-plane integer pixels."""

    reference_depth = state.get("reference_depth_m")
    sample_indices = state.get("sample_reference_pixel_indices")
    write_count = int(state.get("sample_write_count", -1))
    if not isinstance(reference_depth, torch.Tensor) or not isinstance(sample_indices, torch.Tensor):
        raise RuntimeError("Local TacMap integer-pixel layout is not initialized")
    if write_count <= 0 or write_count > int(sample_indices.numel()):
        raise ValueError("Local TacMap integer-pixel write count is invalid")

    depth_flat = torch.nan_to_num(local_depth.detach()).reshape(-1)
    valid_flat = local_valid.detach().to(device=depth_flat.device, dtype=torch.bool).reshape(-1)
    if int(depth_flat.numel()) != int(sample_indices.numel()) or valid_flat.shape != depth_flat.shape:
        raise ValueError("Local TacMap depth/validity does not match the active integer-pixel layout")
    write_indices = sample_indices[:write_count]
    write_depth = torch.where(
        valid_flat[:write_count],
        depth_flat[:write_count],
        torch.zeros_like(depth_flat[:write_count]),
    )
    raster_depth = torch.zeros_like(reference_depth, dtype=torch.float32)
    raster_sampled = torch.zeros_like(reference_depth, dtype=torch.bool)
    raster_depth.reshape(-1).index_copy_(0, write_indices, write_depth)
    raster_sampled.reshape(-1).index_copy_(0, write_indices, valid_flat[:write_count])
    return raster_depth, raster_sampled


def set_visual_local_tacmap_layout(
    sensor,
    state: dict,
    bounds: tuple[float, float, float, float],
) -> None:
    """Install one ROI's camera-Z rays at exact dense-reference integer pixels."""

    (
        starts,
        reference_depth,
        reference_valid,
        pixel_indices,
        write_count,
        roi_valid,
        grid_shape,
    ) = sample_visual_local_reference(state, bounds)
    flat_starts = starts.reshape(-1, 3)
    direction = state["ray_direction_l"].reshape(1, 3).expand(flat_starts.shape[0], -1)
    if int(sensor.ray_starts_att.shape[1]) != VISUAL_LOCAL_TACMAP_RAY_COUNT:
        raise ValueError(
            f"Local TacMap object sensor has {int(sensor.ray_starts_att.shape[1])} rays, "
            f"expected {VISUAL_LOCAL_TACMAP_RAY_COUNT}"
    )
    sensor.ray_starts_att.copy_(flat_starts.unsqueeze(0).expand_as(sensor.ray_starts_att))
    sensor.ray_directions_att.copy_(direction.unsqueeze(0).expand_as(sensor.ray_directions_att))
    reference_points_l = starts + state["ray_direction_l"].reshape(1, 3) * reference_depth.unsqueeze(-1)
    state["sample_bounds"] = tuple(float(value) for value in bounds)
    state["sample_shape"] = (int(VISUAL_LOCAL_TACMAP_RAY_COUNT),)
    state["sample_grid_shape"] = grid_shape
    state["sample_reference_pixel_indices"] = pixel_indices
    state["sample_write_count"] = int(write_count)
    state["sample_roi_valid"] = roi_valid
    state["sample_reference_depth_m"] = reference_depth
    state["sample_reference_valid"] = reference_valid
    state["sample_reference_points_l"] = reference_points_l


def sync_visual_local_tacmap_from_rl(env, state: dict | None = None) -> dict | None:
    """Populate the display state from the exact adaptive Depth used by RL."""

    reference = getattr(env, "_rl_local_tacmap_reference_state", None)
    local_depth = getattr(env, "_rl_local_tacmap_depth_m", None)
    local_debug = getattr(env, "_rl_local_tacmap_debug", None)
    if not (
        isinstance(reference, dict)
        and isinstance(local_depth, torch.Tensor)
        and isinstance(local_debug, dict)
    ):
        return None
    focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
    if local_depth.ndim != 4 or focus_index >= int(local_depth.shape[1]):
        return None
    pixel_indices_all = local_debug.get("pixel_indices")
    reference_points_all = local_debug.get("reference_points_l")
    object_depth_all = local_debug.get("object_depth_m")
    object_valid_all = local_debug.get("object_valid")
    roi_bounds_all = local_debug.get("roi_bounds_m")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            pixel_indices_all,
            reference_points_all,
            object_depth_all,
            object_valid_all,
            roi_bounds_all,
        )
    ):
        return None

    reference_depth = reference["reference_depth_m"][focus_index]
    reference_valid = reference["reference_valid"][focus_index]
    reference_starts = reference["reference_starts_l"][focus_index]
    reference_directions = reference["reference_directions_l"][focus_index]
    reference_xy = reference["reference_xy_camera_m"][focus_index]
    pixel_indices = pixel_indices_all[focus_index].to(device=env.device, dtype=torch.long).reshape(-1)
    depth_flat = local_depth[0, focus_index].detach().to(device=env.device, dtype=torch.float32).reshape(-1)
    write_count = min(int(pixel_indices.numel()), int(depth_flat.numel()))
    if write_count <= 0:
        return None
    pixel_indices = pixel_indices[:write_count]
    sample_reference_depth = reference_depth.reshape(-1).index_select(0, pixel_indices)
    sample_reference_valid = reference_valid.reshape(-1).index_select(0, pixel_indices)
    roi_bounds = roi_bounds_all[focus_index].to(device=env.device, dtype=torch.float32).reshape(4)
    roi_valid = (
        reference_valid
        & (reference_xy[..., 0] >= roi_bounds[0])
        & (reference_xy[..., 0] <= roi_bounds[1])
        & (reference_xy[..., 1] >= roi_bounds[2])
        & (reference_xy[..., 1] <= roi_bounds[3])
    )

    if not isinstance(state, dict) or state.get("source") != "rl_policy_observation":
        boundary_mask = torch.zeros_like(reference_valid, dtype=torch.bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        state = {
            "source": "rl_policy_observation",
            "native_rows": int(TIANJI_TACMAP_RL_ROWS),
            "native_cols": int(TIANJI_TACMAP_RL_COLS),
            "reference_depth_m": reference_depth,
            "reference_valid": reference_valid,
            "reference_starts_l": reference_starts,
            "reference_xy_camera_m": reference_xy,
            "reference_boundary_indices": torch.nonzero(
                (boundary_mask & reference_valid).reshape(-1),
                as_tuple=False,
            ).reshape(-1),
            "ray_direction_l": reference_directions[0, 0],
            "link_name": str(reference["link_names"][focus_index]),
        }

    state["sample_reference_pixel_indices"] = pixel_indices
    state["sample_write_count"] = int(write_count)
    state["sample_reference_depth_m"] = sample_reference_depth
    state["sample_reference_valid"] = sample_reference_valid
    state["sample_reference_points_l"] = reference_points_all[focus_index, :write_count].detach()
    state["display_reference_points_l"] = state["sample_reference_points_l"]
    state["sample_object_depth_m"] = object_depth_all[focus_index, :write_count].detach()
    state["sample_object_valid"] = object_valid_all[focus_index, :write_count].detach()
    state["depth_m"] = depth_flat[:write_count]
    state["valid"] = sample_reference_valid
    state["sample_roi_valid"] = roi_valid
    state["display_valid"] = roi_valid
    state["sample_bounds"] = tuple(float(value) for value in roi_bounds)
    state["display_bounds"] = state["sample_bounds"]
    state["display_depth_m"], state["display_sampled"] = rasterize_visual_local_tacmap_depth(
        state,
        depth_flat[:write_count],
        sample_reference_valid,
    )
    return state


def initialize_visual_local_tacmap_refinement(env) -> dict | None:
    """Ray-cast the dense rubber reference once and initialize the local object-only rays."""

    if not visual_local_tacmap_enabled():
        return None
    rl_state = sync_visual_local_tacmap_from_rl(env)
    if isinstance(rl_state, dict):
        env._rl_visual_local_tacmap_refinement = rl_state
        print(
            "[INFO] Local TacMap visualization and rl_ours_tacmap_policy share the exact "
            "5x25x40 adaptive samples used to reconstruct five full 240x320 metric Depth images; "
            "the grayscale Gaussian blur remains display-only.",
            flush=True,
        )
        return rl_state
    surface_sensor = env.scene.sensors.get(VISUAL_LOCAL_TACMAP_SURFACE_SENSOR)
    object_sensor = env.scene.sensors.get(VISUAL_LOCAL_TACMAP_OBJECT_SENSOR)
    if surface_sensor is None or object_sensor is None:
        raise RuntimeError("Display-only local TacMap sensors were not created")

    finger = str(args_cli.focus_finger)
    reference_rows = int(args_cli.tacmap_display_rows)
    reference_cols = int(args_cli.tacmap_display_cols)
    reference_rows = reference_rows if reference_rows > 0 else 240
    reference_cols = reference_cols if reference_cols > 0 else 320
    camera_rectangles, rectangle_shape, _rectangle_path = _load_camera_ray_rectangles(
        VITAI_MARKER_LAYOUT
    )
    if (reference_rows, reference_cols) != rectangle_shape:
        raise ValueError(
            "Display-only TacMap shape must match the calibrated white rectangle: "
            f"requested={(reference_rows, reference_cols)}, calibrated={rectangle_shape}"
        )
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        camera_origin_l = np.asarray(layout[f"{finger}_camera_origin_link_m"], dtype=np.float32).reshape(3)
        camera_rotation_l = np.asarray(layout[f"{finger}_camera_rotation_link"], dtype=np.float32).reshape(3, 3)
    starts_np, directions_np, storage_row_axis = _camera_plane_ray_grid_from_rectangle(
        camera_rectangles[finger],
        camera_origin_link=camera_origin_l,
        camera_rotation_link=camera_rotation_l,
        rows=reference_rows,
        cols=reference_cols,
    )
    if int(storage_row_axis) != 1:
        raise RuntimeError("White camera rectangle must store rows along camera Y")
    expected_reference_rays = reference_rows * reference_cols
    if int(surface_sensor.ray_starts_att.shape[1]) != expected_reference_rays:
        raise ValueError(
            f"Surface-reference sensor has {int(surface_sensor.ray_starts_att.shape[1])} rays, "
            f"expected {expected_reference_rays}"
        )
    starts_l = torch.as_tensor(starts_np, device=env.device, dtype=torch.float32)
    directions_l = torch.as_tensor(directions_np, device=env.device, dtype=torch.float32)
    starts_camera_np = (starts_np - camera_origin_l.reshape(1, 1, 3)) @ camera_rotation_l
    reference_xy_camera_np = np.asarray(starts_camera_np[..., :2], dtype=np.float32)
    reference_native_row = (
        (np.arange(reference_rows, dtype=np.float32) + 0.5)
        * (float(TIANJI_TACMAP_RL_ROWS) / float(reference_rows))
        - 0.5
    )
    reference_native_col = (
        (np.arange(reference_cols, dtype=np.float32) + 0.5)
        * (float(TIANJI_TACMAP_RL_COLS) / float(reference_cols))
        - 0.5
    )
    full_camera_bounds_m = (
        float(np.min(reference_xy_camera_np[..., 0])),
        float(np.max(reference_xy_camera_np[..., 0])),
        float(np.min(reference_xy_camera_np[..., 1])),
        float(np.max(reference_xy_camera_np[..., 1])),
    )
    surface_sensor.ray_starts_att.copy_(
        starts_l.reshape(1, expected_reference_rays, 3).expand_as(surface_sensor.ray_starts_att)
    )
    surface_sensor.ray_directions_att.copy_(
        directions_l.reshape(1, expected_reference_rays, 3).expand_as(surface_sensor.ray_directions_att)
    )
    surface_sensor.update(0.0, force_recompute=True)
    reference_raw = surface_sensor.data.output["distance_along_normal_raw"][0].reshape(
        reference_rows,
        reference_cols,
    )
    reference_depth = torch.nan_to_num(reference_raw.detach().to(device=env.device, dtype=torch.float32))
    reference_valid = surface_sensor.ray_hit_valid[0].detach().reshape(reference_rows, reference_cols)
    reference_valid = reference_valid & torch.isfinite(reference_raw) & (reference_depth > 0.0)
    reference_boundary_mask = torch.zeros_like(reference_valid, dtype=torch.bool)
    reference_boundary_mask[0, :] = True
    reference_boundary_mask[-1, :] = True
    reference_boundary_mask[:, 0] = True
    reference_boundary_mask[:, -1] = True
    reference_boundary_indices = torch.nonzero(
        (reference_boundary_mask & reference_valid).reshape(-1),
        as_tuple=False,
    ).reshape(-1)
    state = {
        "native_rows": int(TIANJI_TACMAP_RL_ROWS),
        "native_cols": int(TIANJI_TACMAP_RL_COLS),
        "reference_depth_m": reference_depth,
        "reference_valid": reference_valid,
        "reference_boundary_indices": reference_boundary_indices,
        "reference_starts_l": starts_l,
        "reference_xy_camera_m": torch.as_tensor(
            reference_xy_camera_np,
            device=env.device,
            dtype=torch.float32,
        ),
        "reference_xy_camera_np": reference_xy_camera_np,
        "reference_native_row": reference_native_row,
        "reference_native_col": reference_native_col,
        "full_camera_bounds_m": full_camera_bounds_m,
        "ray_direction_l": directions_l[0, 0],
        "link_name": str(TIANJI_TACMAP_LINK_ORDER[FINGER_CHOICES.index(finger)]),
        "roi_locked": False,
        "pending_bounds": None,
        "pending_count": 0,
    }
    set_visual_local_tacmap_layout(object_sensor, state, full_camera_bounds_m)
    write_count = int(state["sample_write_count"])
    state["display_bounds"] = state["sample_bounds"]
    state["display_reference_points_l"] = state["sample_reference_points_l"][:write_count]
    state["depth_m"] = torch.zeros_like(state["sample_reference_depth_m"][:write_count])
    state["valid"] = state["sample_reference_valid"][:write_count]
    state["sample_object_depth_m"] = torch.zeros_like(state["sample_reference_depth_m"][:write_count])
    state["sample_object_valid"] = torch.zeros_like(
        state["sample_reference_valid"][:write_count],
        dtype=torch.bool,
    )
    state["display_depth_m"], state["display_sampled"] = rasterize_visual_local_tacmap_depth(
        state,
        torch.zeros_like(state["sample_reference_depth_m"]),
        state["sample_reference_valid"],
    )
    state["display_valid"] = state["sample_roi_valid"]
    valid_count = int(reference_valid.count_nonzero().detach().cpu())
    print(
        "[INFO] Display-only local TacMap refinement initialized: "
        f"surface_reference={reference_rows}x{reference_cols} ({valid_count}/{expected_reference_rays} valid, once), "
        f"runtime_object_rays={VISUAL_LOCAL_TACMAP_RAY_COUNT}, "
        f"depth_raster=yellow_plane_{reference_rows}x{reference_cols}_direct_integer_pixels; "
        "metric_nearest=off; RL observation remains 5x"
        f"{TIANJI_TACMAP_RL_ROWS}x{TIANJI_TACMAP_RL_COLS}.",
        flush=True,
    )
    env._rl_visual_local_tacmap_refinement = state
    return state


def update_visual_local_tacmap_refinement(env) -> None:
    """Update the 1000 local rays every environment step, including same-step ROI relayouts."""

    state = getattr(env, "_rl_visual_local_tacmap_refinement", None)
    if not isinstance(state, dict):
        return
    if state.get("source") == "rl_policy_observation":
        synced = sync_visual_local_tacmap_from_rl(env, state)
        if isinstance(synced, dict):
            env._rl_visual_local_tacmap_refinement = synced
        return
    sensor = env.scene.sensors.get(VISUAL_LOCAL_TACMAP_OBJECT_SENSOR)
    if sensor is None:
        return

    layout_changed = False
    next_bounds = visual_local_tacmap_contact_roi(env, state)
    if next_bounds is None:
        state["pending_bounds"] = None
        state["pending_count"] = 0
    else:
        current_bounds = tuple(float(value) for value in state["sample_bounds"])
        guard_m = max(0.0, float(args_cli.tacmap_local_roi_guard_mm)) * 1.0e-3
        bounds_changed = any(
            abs(float(candidate) - float(current)) > guard_m
            for candidate, current in zip(next_bounds, current_bounds, strict=True)
        )
        requires_layout = not bool(state.get("roi_locked", False)) or bounds_changed
        if not requires_layout:
            state["pending_bounds"] = None
            state["pending_count"] = 0
        else:
            pending_bounds = state.get("pending_bounds")
            stable_tolerance_m = max(1.0e-6, min(0.25 * guard_m, 0.1e-3))
            same_candidate = isinstance(pending_bounds, tuple) and all(
                abs(float(a) - float(b)) <= stable_tolerance_m
                for a, b in zip(pending_bounds, next_bounds, strict=True)
            )
            if same_candidate:
                state["pending_count"] = int(state.get("pending_count", 0)) + 1
            else:
                state["pending_bounds"] = tuple(float(value) for value in next_bounds)
                state["pending_count"] = 1
            if int(state["pending_count"]) >= max(1, int(args_cli.tacmap_local_roi_stable_frames)):
                set_visual_local_tacmap_layout(sensor, state, tuple(state["pending_bounds"]))
                state["roi_locked"] = True
                state["pending_bounds"] = None
                state["pending_count"] = 0
                layout_changed = True

    if layout_changed:
        outdated = getattr(sensor, "_is_outdated", None)
        if isinstance(outdated, torch.Tensor):
            outdated[:] = True
        sensor.update(0.0, force_recompute=True)

    raw = sensor.data.output["distance_along_normal_raw"][0].reshape(-1)
    object_valid = sensor.ray_hit_valid[0].detach().reshape(-1)
    object_depth = torch.nan_to_num(raw.detach().to(device=env.device, dtype=torch.float32))
    reference_depth = state["sample_reference_depth_m"]
    reference_valid = state["sample_reference_valid"]
    penetration = torch.where(
        object_valid & reference_valid,
        torch.clamp(reference_depth - object_depth, min=0.0),
        torch.zeros_like(reference_depth),
    )
    penetration = torch.where(penetration > 1.0e-6, penetration, torch.zeros_like(penetration))
    write_count = int(state["sample_write_count"])
    state["depth_m"] = penetration[:write_count]
    state["sample_object_depth_m"] = object_depth[:write_count]
    state["sample_object_valid"] = object_valid[:write_count]
    # A missing object hit is a valid zero-contact observation. Only the static
    # rubber-reference validity limits which local pixels may replace the display.
    state["valid"] = reference_valid[:write_count]
    state["display_bounds"] = state["sample_bounds"]
    state["display_reference_points_l"] = state["sample_reference_points_l"][:write_count]
    state["display_depth_m"], state["display_sampled"] = rasterize_visual_local_tacmap_depth(
        state,
        penetration,
        reference_valid,
    )
    state["display_valid"] = state["sample_roi_valid"]


def gaussian_blur_visual_tacmap(
    gray: torch.Tensor,
    sampled: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SharpA blur, normalizing sparse direct-pixel samples when provided."""

    if gray.ndim != 2:
        raise ValueError("Visual TacMap Gaussian blur expects one HxW image")
    if sampled is not None and sampled.shape != gray.shape:
        raise ValueError("Visual TacMap Gaussian sample mask must match the HxW image")
    resolution_scale = max(
        1.0,
        float(gray.shape[0]) / 320.0,
        float(gray.shape[1]) / 240.0,
    )
    base_radius = int(VISUAL_LOCAL_TACMAP_GAUSSIAN_KERNEL_SIZE) // 2
    kernel_size = 2 * max(1, int(round(base_radius * resolution_scale))) + 1
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Visual TacMap Gaussian kernel size must be a positive odd integer")
    sigma = float(VISUAL_LOCAL_TACMAP_GAUSSIAN_SIGMA) * resolution_scale
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Visual TacMap Gaussian sigma must be positive and finite")

    radius = kernel_size // 2
    coordinates = torch.arange(-radius, radius + 1, device=gray.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel_1d /= torch.sum(kernel_1d)
    kernel_2d = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, kernel_size, kernel_size)
    image = gray.to(dtype=torch.float32).reshape(1, 1, int(gray.shape[0]), int(gray.shape[1]))
    pad_mode = "reflect" if int(gray.shape[0]) > radius and int(gray.shape[1]) > radius else "replicate"
    if sampled is None:
        image = torch.nn.functional.pad(image, (radius, radius, radius, radius), mode=pad_mode)
        blurred = torch.nn.functional.conv2d(image, kernel_2d)[0, 0]
    else:
        sample_weights = sampled.to(device=gray.device, dtype=torch.float32).reshape_as(image)
        weighted_image = torch.nn.functional.pad(
            image * sample_weights,
            (radius, radius, radius, radius),
            mode=pad_mode,
        )
        padded_weights = torch.nn.functional.pad(
            sample_weights,
            (radius, radius, radius, radius),
            mode=pad_mode,
        )
        numerator = torch.nn.functional.conv2d(weighted_image, kernel_2d)[0, 0]
        denominator = torch.nn.functional.conv2d(padded_weights, kernel_2d)[0, 0]
        blurred = torch.where(
            denominator > 1.0e-8,
            numerator / torch.clamp(denominator, min=1.0e-8),
            torch.zeros_like(numerator),
        )
    return torch.clamp(torch.round(blurred), 0.0, 255.0).to(dtype=torch.uint8)


def overlay_visual_local_tacmap_refinement(
    strip: torch.Tensor,
    state: dict | None,
    *,
    pane_index: int,
    pane_count: int,
    display_max_m: float,
) -> torch.Tensor:
    """Replace the expanded ROI in one rendered pane with the local-ray depth."""

    if not isinstance(state, dict):
        return strip
    raster_depth = state.get("display_depth_m")
    raster_valid = state.get("display_valid")
    raster_sampled = state.get("display_sampled")
    if not (
        isinstance(raster_depth, torch.Tensor)
        and isinstance(raster_valid, torch.Tensor)
        and isinstance(raster_sampled, torch.Tensor)
    ):
        return strip
    pane_count = max(1, int(pane_count))
    pane_index = max(0, min(int(pane_index), pane_count - 1))
    pane_width = int(strip.shape[1]) // pane_count
    if pane_width <= 0:
        return strip

    target_rows = int(strip.shape[0])
    target_cols = pane_width
    depth_display = display_tacmap_maps(raster_depth).to(device=strip.device, dtype=torch.float32)
    valid_display = display_tacmap_maps(raster_valid).to(device=strip.device, dtype=torch.bool)
    sampled_display = display_tacmap_maps(raster_sampled).to(device=strip.device, dtype=torch.bool)
    if tuple(depth_display.shape) != (target_rows, target_cols):
        depth_display = torch.nn.functional.interpolate(
            depth_display.unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols),
            mode="nearest",
        )[0, 0]
        valid_display = torch.nn.functional.interpolate(
            valid_display.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols),
            mode="nearest",
        )[0, 0] > 0.5
        sampled_display = torch.nn.functional.interpolate(
            sampled_display.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols),
            mode="nearest",
        )[0, 0] > 0.5
    replace = valid_display
    gamma = max(1.0e-6, float(args_cli.tacmap_gamma))
    if float(display_max_m) > 0.0:
        normalized = torch.clamp(depth_display / float(display_max_m), 0.0, 1.0)
    else:
        local_max = torch.amax(depth_display)
        normalized = torch.where(
            local_max > 0.0,
            depth_display / torch.clamp(local_max, min=1.0e-8),
            depth_display,
        )
    local_gray = torch.clamp(torch.pow(normalized, gamma) * 255.0, 0.0, 255.0).to(torch.uint8)
    local_gray = gaussian_blur_visual_tacmap(local_gray, sampled_display)
    start_col = pane_index * pane_width
    pane = strip[:, start_col : start_col + pane_width]
    original = pane.clone()
    pane[replace] = local_gray[replace].unsqueeze(-1).expand(-1, 3)
    red_ring = (original[..., 0] == 255) & (original[..., 1] == 0) & (original[..., 2] == 0)
    pane[red_ring] = original[red_ring]
    return strip


def render_visual_taxim_rgb(
    env,
    component_cfg: dict | None,
) -> tuple[np.ndarray | None, str]:
    """Display the exact Taxim RGB tensor fed into the frozen RL ResNet."""

    if component_cfg is None:
        return None, "taxim_rgb=off"

    cfg = dict(component_cfg)
    render_rows = max(1, int(cfg.get("taxim_rgb_render_rows", 240)))
    render_cols = max(1, int(cfg.get("taxim_rgb_render_cols", 320)))
    finger_count = len(FINGER_CHOICES)
    focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))

    # The policy observation wrapper calls this same raw-RGB producer before
    # its frozen ResNet.  Thus the preview is the encoder input, while the
    # ObservationManager exposes only the resulting 128-D embedding.
    policy_rgb = ours_rl_taxim_rgb_obs(env, cfg)
    expected_shape = (env.num_envs, finger_count * 3 * render_rows * render_cols)
    if tuple(policy_rgb.shape) != expected_shape:
        raise RuntimeError(
            f"RL Taxim RGB visualization received {tuple(policy_rgb.shape)}, expected {expected_shape}"
        )
    rgb_chw = policy_rgb.reshape(
        env.num_envs,
        finger_count,
        3,
        render_rows,
        render_cols,
    )[0, focus_index]
    rgb_hwc_u8 = torch.clamp(
        torch.round(rgb_chw.permute(1, 2, 0) * 255.0),
        0.0,
        255.0,
    ).to(dtype=torch.uint8)
    env._rl_visual_taxim_rgb_policy_chw = rgb_chw
    env._rl_visual_taxim_composited_rgb = rgb_hwc_u8
    rgb = to_numpy_uint8_rgb(rgb_hwc_u8)

    dense_depth = getattr(env, "_rl_ours_dense_tacmap_depth_m", None)
    if (
        isinstance(dense_depth, torch.Tensor)
        and dense_depth.ndim == 4
        and int(dense_depth.shape[0]) >= 1
        and focus_index < int(dense_depth.shape[1])
    ):
        depth_m = dense_depth[0, focus_index]
        env._rl_visual_taxim_input_depth_m = depth_m
        max_depth_mm = float(torch.amax(depth_m).detach().cpu()) * 1000.0
        active_pixels = int(torch.count_nonzero(depth_m > 0.0).detach().cpu())
    else:
        max_depth_mm = 0.0
        active_pixels = 0

    background_path = str(cfg.get("taxim_rgb_background_path", "")).strip()
    background_mode = Path(background_path).name if background_path else "taxim"
    return (
        rgb,
        f"taxim_rgb={render_rows}x{render_cols} "
        f"source:rl_taxim_pre_resnet_input background:{background_mode} "
        f"depth:{max_depth_mm:.3f}mm active:{active_pixels}",
    )


def build_joint_tacmap_interpolation_layout(
    marker_coords: np.ndarray,
    marker_valid: np.ndarray,
    *,
    source_rows: int,
    source_cols: int,
    target_rows: int,
    target_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute one piecewise-linear interpolation over base and Marker samples."""

    from scipy.spatial import Delaunay

    coords = np.asarray(marker_coords, dtype=np.float64)
    valid = np.asarray(marker_valid, dtype=bool)
    if coords.ndim != 3 or coords.shape[-1] != 2 or valid.shape != coords.shape[:-1]:
        raise ValueError("Marker coordinates/validity must have shapes FxMx2 and FxM")
    source_rows = max(1, int(source_rows))
    source_cols = max(1, int(source_cols))
    target_rows = max(1, int(target_rows))
    target_cols = max(1, int(target_cols))

    base_rows, base_cols = np.meshgrid(
        np.arange(source_rows, dtype=np.float64),
        np.arange(source_cols, dtype=np.float64),
        indexing="ij",
    )
    base_coords = np.column_stack((base_rows.reshape(-1), base_cols.reshape(-1)))
    base_count = int(len(base_coords))
    target_y = (np.arange(target_rows, dtype=np.float64) + 0.5) * source_rows / target_rows - 0.5
    target_x = (np.arange(target_cols, dtype=np.float64) + 0.5) * source_cols / target_cols - 0.5
    target_y = np.clip(target_y, 0.0, float(source_rows - 1))
    target_x = np.clip(target_x, 0.0, float(source_cols - 1))
    target_grid_y, target_grid_x = np.meshgrid(target_y, target_x, indexing="ij")
    targets = np.column_stack((target_grid_y.reshape(-1), target_grid_x.reshape(-1)))

    index_sets: list[np.ndarray] = []
    weight_sets: list[np.ndarray] = []
    pixel_valid_sets: list[np.ndarray] = []
    for finger_index in range(int(coords.shape[0])):
        valid_ids = np.flatnonzero(valid[finger_index] & np.isfinite(coords[finger_index]).all(axis=-1))
        samples = np.concatenate((base_coords, coords[finger_index, valid_ids]), axis=0)
        source_ids = np.concatenate((np.arange(base_count), base_count + valid_ids), axis=0)
        triangulation = Delaunay(samples)
        simplex = triangulation.find_simplex(targets, tol=1.0e-10)
        safe_simplex = np.maximum(simplex, 0)
        transform = triangulation.transform[safe_simplex]
        delta = targets - transform[:, 2]
        barycentric_first = np.einsum("nij,nj->ni", transform[:, :2], delta)
        barycentric = np.column_stack(
            (barycentric_first, 1.0 - np.sum(barycentric_first, axis=1))
        )
        pixel_valid = (
            (simplex >= 0)
            & np.isfinite(barycentric).all(axis=-1)
            & np.all(barycentric >= -1.0e-5, axis=-1)
        )
        weights = np.clip(barycentric, 0.0, 1.0)
        weights /= np.maximum(np.sum(weights, axis=-1, keepdims=True), 1.0e-12)
        indices = source_ids[triangulation.simplices[safe_simplex]]
        indices[~pixel_valid] = 0
        weights[~pixel_valid] = 0.0
        index_sets.append(indices.reshape(target_rows, target_cols, 3).astype(np.int64))
        weight_sets.append(weights.reshape(target_rows, target_cols, 3).astype(np.float32))
        pixel_valid_sets.append(pixel_valid.reshape(target_rows, target_cols))

    return (
        np.stack(index_sets, axis=0),
        np.stack(weight_sets, axis=0),
        np.stack(pixel_valid_sets, axis=0),
    )


def tacmap_marker_joint_inputs(
    env,
    *,
    source_rows: int,
    source_cols: int,
    target_rows: int,
    target_cols: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Return one base/Marker interpolation layout for display only."""

    marker_depth_cache = getattr(env, "_rl_hydroshear_marker_depth_m", None)
    marker_valid_cache = getattr(env, "_rl_hydroshear_marker_depth_valid", None)
    marker_ready = (
        bool(args_cli.tacmap_marker_depth_fusion)
        and isinstance(marker_depth_cache, torch.Tensor)
        and isinstance(marker_valid_cache, torch.Tensor)
        and marker_depth_cache.ndim == 4
        and marker_valid_cache.shape == marker_depth_cache.shape
        and int(marker_depth_cache.shape[0]) > 0
    )
    if not marker_ready:
        return None, None, None, None, None
    depth = marker_depth_cache
    valid = marker_valid_cache
    assert isinstance(depth, torch.Tensor) and isinstance(valid, torch.Tensor)
    coords_np, layout_valid_np = calibrated_marker_tacmap_grid_layout()
    finger_count = min(int(depth.shape[1]), int(coords_np.shape[0]))
    depth_by_finger = depth[0, :finger_count].flatten(start_dim=1)
    valid_by_finger = valid[0, :finger_count].flatten(start_dim=1)
    marker_count = min(int(depth_by_finger.shape[1]), int(coords_np.shape[1]))
    coords_np = coords_np[:finger_count, :marker_count]
    layout_valid_np = layout_valid_np[:finger_count, :marker_count]
    depth_by_finger = depth_by_finger[:, :marker_count]
    valid_by_finger = valid_by_finger[:, :marker_count]
    if finger_count <= 0:
        return None, None, None, None, None
    static_cache_key = (
        str(depth.device),
        int(source_rows),
        int(source_cols),
        int(target_rows),
        int(target_cols),
        int(args_cli.tacmap_display_rot90) % 4,
        bool(args_cli.tacmap_display_flip_x),
        bool(args_cli.tacmap_display_flip_y),
        finger_count,
        marker_count,
    )
    static_layout_cache = getattr(env, "_rl_visual_marker_tacmap_static_layout", None)
    if (
        not isinstance(static_layout_cache, tuple)
        or len(static_layout_cache) != 5
        or static_layout_cache[0] != static_cache_key
    ):
        coords = display_tacmap_point_coords(
            torch.as_tensor(coords_np, dtype=torch.float32),
            source_rows=TIANJI_TACMAP_RL_ROWS,
            source_cols=TIANJI_TACMAP_RL_COLS,
        )
        static_valid = layout_valid_np
        indices_np, weights_np, pixel_valid_np = build_joint_tacmap_interpolation_layout(
            coords.numpy(),
            static_valid,
            source_rows=source_rows,
            source_cols=source_cols,
            target_rows=target_rows,
            target_cols=target_cols,
        )
        interpolation_layout = (
            torch.as_tensor(indices_np, device=depth.device, dtype=torch.long),
            torch.as_tensor(weights_np, device=depth.device, dtype=torch.float32),
            torch.as_tensor(pixel_valid_np, device=depth.device, dtype=torch.bool),
            torch.as_tensor(static_valid, device=depth.device, dtype=torch.bool),
        )
        static_layout_cache = (static_cache_key, *interpolation_layout)
        env._rl_visual_marker_tacmap_static_layout = static_layout_cache
    _, static_indices, static_weights, static_pixel_valid, static_valid = static_layout_cache
    depth_flat = torch.nan_to_num(depth_by_finger.detach())
    valid_flat = valid_by_finger.detach()
    valid_flat = valid_flat & static_valid
    return depth_flat, valid_flat, static_indices, static_weights, static_pixel_valid


def tacmap_display_target_size() -> tuple[int, int] | None:
    rows = int(args_cli.tacmap_display_rows)
    cols = int(args_cli.tacmap_display_cols)
    if rows <= 0 or cols <= 0:
        return None
    return (rows, cols)


def tacmap_display_surface_geometry(env, *, force: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not bool(force) and str(args_cli.tacmap_resize_mode).lower() != "surface":
        return None, None
    points = getattr(env, "_rl_tacmap_surface_points_w", None)
    valid = getattr(env, "_rl_tacmap_surface_valid", None)
    if not (isinstance(points, torch.Tensor) and isinstance(valid, torch.Tensor)):
        return None, None
    if points.ndim != 5 or valid.ndim != 4 or points.shape[0] < 1 or valid.shape[0] < 1:
        return None, None
    points0 = torch.nan_to_num(points[0].detach().to(device=env.device, dtype=torch.float32))
    valid0 = valid[0].detach().to(device=env.device, dtype=torch.bool)
    return display_tacmap_vec_maps(points0), display_tacmap_maps(valid0)


def tile_rgb(panes, cols: int, gap: int = 4, *, preserve_scale: bool = False) -> np.ndarray:
    arrays = [to_numpy_uint8_rgb(pane)[:, :, :3] for pane in panes if pane is not None]
    if not arrays:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, int(cols))
    rows = []
    gap_col = np.full((1, gap, 3), 32, dtype=np.uint8)
    for i in range(0, len(arrays), cols):
        row_items = arrays[i : i + cols]
        height = max(int(item.shape[0]) for item in row_items)
        if preserve_scale:
            row_items = [
                np.pad(
                    item,
                    ((0, height - int(item.shape[0])), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=32,
                )
                for item in row_items
            ]
        else:
            row_items = [resize_nearest_to_shape(item, height, int(item.shape[1])) for item in row_items]
        gap_col = np.full((height, gap, 3), 32, dtype=np.uint8)
        row = row_items[0]
        for item in row_items[1:]:
            row = np.concatenate((row, gap_col, item), axis=1)
        rows.append(row)
    width = max(int(row.shape[1]) for row in rows)
    padded = []
    for row in rows:
        if int(row.shape[1]) < width:
            pad = np.full((int(row.shape[0]), width - int(row.shape[1]), 3), 32, dtype=np.uint8)
            row = np.concatenate((row, pad), axis=1)
        padded.append(row)
    gap_row = np.full((gap, width, 3), 32, dtype=np.uint8)
    out = padded[0]
    for row in padded[1:]:
        out = np.concatenate((out, gap_row, row), axis=0)
    return out


def hydroshear_output_from_env(env):
    return getattr(env, "_rl_hydroshear_output", None)


def hydroshear_object_sample_debug_points(env) -> np.ndarray:
    output = hydroshear_output_from_env(env)
    point_sets = None if output is None else getattr(output, "debug_all_object_sample_points_w", None)
    if not point_sets and output is not None:
        point_sets = getattr(output, "debug_object_sample_points_w", None)
    if point_sets:
        index = min(FINGER_CHOICES.index(str(args_cli.focus_finger)), len(point_sets) - 1)
        points = np.asarray(point_sets[index], dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=-1)]
        if len(points):
            return points

    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    points_l = None if adapter is None else getattr(adapter, "_object_sample_points_l", None)
    if not isinstance(points_l, torch.Tensor) or points_l.ndim != 2 or points_l.shape[-1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    obj: RigidObject = env.scene["object"]
    pos_w = obj.data.root_pos_w[0]
    quat_w = obj.data.root_quat_w[0].unsqueeze(0).expand(points_l.shape[0], -1)
    points_w = pos_w.unsqueeze(0) + quat_apply(quat_w, points_l.to(device=env.device, dtype=torch.float32))
    points = points_w.detach().cpu().numpy()
    return points[np.isfinite(points).all(axis=-1)]


def hydroshear_selected_roi_debug_arrays(
    env,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return env 0's exact HydroShear ROI slots and their current contact class.

    The adapter stores the selected Poisson sample IDs in flattened
    ``[env, finger]`` order after every HydroSoft update.  Reading those IDs
    here keeps this visualization identical to the samples used by the policy
    without recomputing or mutating the ROI selection.
    """

    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    points_l = None if adapter is None else getattr(adapter, "_object_sample_points_l", None)
    selected_ids = None if adapter is None else getattr(adapter, "_batch_prev_sample_ids", None)
    selected_valid = None if adapter is None else getattr(adapter, "_batch_state_valid", None)
    selected_sdf = None if adapter is None else getattr(adapter, "_batch_prev_sdf", None)
    if not (
        isinstance(points_l, torch.Tensor)
        and isinstance(selected_ids, torch.Tensor)
        and isinstance(selected_valid, torch.Tensor)
        and isinstance(selected_sdf, torch.Tensor)
        and points_l.ndim == 2
        and points_l.shape[-1] == 3
        and selected_ids.ndim == 2
        and selected_valid.shape == selected_ids.shape
        and selected_sdf.shape == selected_ids.shape
        and int(points_l.shape[0]) > 0
    ):
        return None

    slot = FINGER_CHOICES.index(str(args_cli.focus_finger))
    if int(selected_ids.shape[0]) <= slot:
        return None
    ids = selected_ids[slot].to(device=env.device, dtype=torch.long)
    valid = (
        selected_valid[slot].to(device=env.device, dtype=torch.bool)
        & (ids >= 0)
        & (ids < int(points_l.shape[0]))
    )
    ids_clamped = torch.clamp(ids, 0, int(points_l.shape[0]) - 1)
    selected_points_l = points_l.to(device=env.device, dtype=torch.float32)[ids_clamped]

    obj: RigidObject = env.scene["object"]
    object_pos_w = obj.data.root_pos_w[0].to(device=env.device, dtype=torch.float32)
    object_quat_w = obj.data.root_quat_w[0].to(device=env.device, dtype=torch.float32)
    selected_points_w = object_pos_w.unsqueeze(0) + quat_apply(
        object_quat_w.unsqueeze(0).expand(selected_points_l.shape[0], -1),
        selected_points_l,
    )
    contact = valid & (selected_sdf[slot].to(device=env.device, dtype=torch.float32) < 0.0)
    packed = torch.cat(
        (
            selected_points_w,
            valid[:, None].to(dtype=torch.float32),
            contact[:, None].to(dtype=torch.float32),
        ),
        dim=-1,
    ).detach().cpu().numpy()
    colors = np.tile(np.asarray((1.0, 0.82, 0.0), dtype=np.float32), (packed.shape[0], 1))
    colors[packed[:, 4] > 0.5] = np.asarray((1.0, 0.05, 0.05), dtype=np.float32)
    return packed[:, :3], packed[:, 3] > 0.5, colors


def hydroshear_global_tacmap_surface_debug_arrays(
    env,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return the focus finger's exact 32x24 global TacMap surface hits."""

    points = getattr(env, "_rl_tacmap_surface_points_w", None)
    valid = getattr(env, "_rl_tacmap_surface_valid", None)
    if not (
        isinstance(points, torch.Tensor)
        and isinstance(valid, torch.Tensor)
        and points.ndim == 5
        and valid.ndim == 4
        and points.shape[:-1] == valid.shape
    ):
        return None
    slot = FINGER_CHOICES.index(str(args_cli.focus_finger))
    if int(points.shape[0]) <= 0 or int(points.shape[1]) <= slot:
        return None
    points_focus = points[0, slot].to(device=env.device, dtype=torch.float32)
    valid_focus = valid[0, slot].to(device=env.device, dtype=torch.bool)
    if tuple(points_focus.shape[:2]) != (
        int(TIANJI_TACMAP_RL_ROWS),
        int(TIANJI_TACMAP_RL_COLS),
    ):
        return None
    packed = torch.cat(
        (points_focus, valid_focus[..., None].to(dtype=torch.float32)),
        dim=-1,
    ).detach().cpu().numpy()

    rows, cols = packed.shape[:2]
    colors = np.tile(np.asarray((0.0, 0.85, 1.0), dtype=np.float32), (rows, cols, 1))
    boundary = np.zeros((rows, cols), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    colors[boundary] = np.asarray((0.95, 0.15, 1.0), dtype=np.float32)
    return packed[..., :3], packed[..., 3] > 0.5, colors


def image_from_sensor_batch(images, sensor_index: int) -> np.ndarray | None:
    if images is None:
        return None
    arr = images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    if arr.ndim == 4:
        if int(arr.shape[0]) <= 0:
            return None
        idx = max(0, min(int(sensor_index), int(arr.shape[0]) - 1))
        arr = arr[idx]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        pass
    elif arr.ndim == 3 and int(arr.shape[0]) > 0:
        arr = arr[0]
    else:
        return None
    return to_numpy_uint8_rgb(arr)


def hydroshear_marker_pane(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    image = to_numpy_uint8_rgb(image)
    background_path = globals().get("VITAI_MARKER_BACKGROUND")
    if background_path is not None:
        cache_key = str(background_path)
        cached = getattr(hydroshear_marker_pane, "_background_cache", None)
        if cached is None or cached[0] != cache_key:
            import cv2

            background_bgr = cv2.imread(cache_key, cv2.IMREAD_COLOR)
            background = (
                None
                if background_bgr is None
                else np.ascontiguousarray(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB))
            )
            cached = (cache_key, background)
            hydroshear_marker_pane._background_cache = cached
        background = cached[1]
        if background is not None:
            image_height, image_width = image.shape[:2]
            background_height, background_width = background.shape[:2]
            if image_height >= background_height and image_width >= background_width:
                pad_y = image_height - background_height
                pad_x = image_width - background_width
                background = np.pad(
                    background,
                    (
                        (pad_y // 2, pad_y - pad_y // 2),
                        (pad_x // 2, pad_x - pad_x // 2),
                        (0, 0),
                    ),
                    mode="edge",
                )
            elif (image_height, image_width) != (background_height, background_width):
                import cv2

                background = cv2.resize(
                    background,
                    (image_width, image_height),
                    interpolation=cv2.INTER_AREA,
                )
            foreground = np.any(image != 0, axis=-1)
            image = np.where(foreground[..., None], image, background)
    return np.ascontiguousarray(image)


def build_hydroshear_marker_image(env) -> np.ndarray | None:
    if not bool(args_cli.show_hydroshear_marker):
        return None
    output = hydroshear_output_from_env(env)
    if output is None:
        return None
    images = getattr(output, "marker_images", None)
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    marker_flow_batch = getattr(output, "marker_flow", None)
    marker_flow = None
    if getattr(marker_flow_batch, "ndim", 0) == 4 and int(marker_flow_batch.shape[0]) > focus_idx:
        marker_flow_focus = marker_flow_batch[focus_idx]
        marker_flow = (
            marker_flow_focus.detach().cpu().numpy()
            if callable(getattr(marker_flow_focus, "detach", None))
            else np.asarray(marker_flow_focus)
        )
    marker_layout = getattr(env, "_brainco_rl_hydroshear_marker_layout", None)
    marker_valid = None if not isinstance(marker_layout, tuple) else np.asarray(marker_layout[3], dtype=bool)
    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    renderer = getattr(adapter, "_render_vector_field_image", None)

    image = image_from_sensor_batch(images, focus_idx)
    visible_flow = None
    if (
        callable(renderer)
        and isinstance(marker_flow, np.ndarray)
        and marker_flow.ndim == 3
        and marker_flow.shape[0] == 2
        and marker_valid is not None
        and focus_idx < marker_valid.shape[0]
        and marker_valid.shape[1] == marker_flow.shape[1]
    ):
        visible_flow = np.asarray(marker_flow[:, marker_valid[focus_idx]], dtype=np.float32).copy()
        motion_scale = float(getattr(args_cli, "hydroshear_marker_motion_scale", 1.0))
        visible_flow[1] = visible_flow[0] + (visible_flow[1] - visible_flow[0]) * motion_scale
        image = renderer(
            visible_flow,
            color=(255, 255, 0),
            draw_points=False,
            padding_px=0,
        )
    image = hydroshear_marker_pane(image)
    if image is not None and visible_flow is not None:
        import cv2

        image = np.ascontiguousarray(image).copy()
        marker_radius = max(
            1,
            int(getattr(getattr(adapter, "cfg", None), "marker_radius", 3)) - 1,
        )
        for start in visible_flow[0]:
            center = (int(round(float(start[0]))), int(round(float(start[1]))))
            cv2.circle(image, center, marker_radius, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    return image


def hydroshear_marker_stats(env) -> str:
    output = hydroshear_output_from_env(env)
    if output is None:
        return "hydroshear=waiting" if bool(args_cli.show_hydroshear_marker) else ""
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    attr_names = ("active_markers", "active_object_samples", "max_depth_mm", "max_marker_motion_px")
    selected: list[torch.Tensor] = []
    for attr_name in attr_names:
        value = getattr(output, attr_name, None)
        if not isinstance(value, torch.Tensor) or value.numel() <= 0:
            selected = []
            break
        flat = value.detach().reshape(-1)
        selected.append(flat[max(0, min(focus_idx, int(flat.numel()) - 1))].to(dtype=torch.float32))
    if selected:
        active_markers, active_objects, max_depth_mm, max_motion_px = torch.stack(selected).cpu().tolist()
    else:
        fallback: list[float] = []
        for attr_name in attr_names:
            arr = np.asarray(getattr(output, attr_name, ()), dtype=np.float32).reshape(-1)
            fallback.append(float(arr[max(0, min(focus_idx, arr.size - 1))]) if arr.size else 0.0)
        active_markers, active_objects, max_depth_mm, max_motion_px = fallback

    return (
        f"hydroshear_active={int(active_markers)} "
        f"hydroshear_obj={int(active_objects)} "
        f"hydroshear_depth={max_depth_mm:.3f}mm "
        f"hydroshear_marker={max_motion_px:.3f}px "
        f"hydroshear_global_tacmap={int(getattr(env, '_rl_hydroshear_tacmap_point_count', 0))}/"
        f"{int(TIANJI_TACMAP_RL_ROWS) * int(TIANJI_TACMAP_RL_COLS)} "
        f"hydroshear_roi={int(getattr(env, '_rl_hydroshear_roi_selected_count', 0))}/"
        f"{int(getattr(env, '_rl_hydroshear_roi_capacity', 0))} "
        f"hydroshear_roi_contact={int(getattr(env, '_rl_hydroshear_roi_contact_count', 0))} "
        f"hydroshear_roi_lines={int(getattr(env, '_rl_hydroshear_roi_line_count', 0))}"
    )


def batched_tacmap_image(
    maps: torch.Tensor,
    *,
    surface_points: torch.Tensor | None,
    surface_valid: torch.Tensor | None,
    marker_depth: torch.Tensor | None,
    marker_depth_valid: torch.Tensor | None,
    joint_source_indices: torch.Tensor | None,
    joint_source_weights: torch.Tensor | None,
    joint_pixel_valid: torch.Tensor | None,
    display_max_m: float,
    target_size: tuple[int, int] | None,
    local_refinement: dict | None = None,
    local_refinement_pane: int = 0,
) -> np.ndarray:
    """Render local-ray depth, or the legacy TacMap display when disabled."""

    pane_count = max(1, int(maps.shape[0]))
    if isinstance(local_refinement, dict):
        if target_size is not None and int(target_size[0]) > 0 and int(target_size[1]) > 0:
            pane_rows, pane_cols = int(target_size[0]), int(target_size[1])
        else:
            scale = max(1, int(args_cli.tacmap_scale))
            pane_rows = int(maps.shape[-2]) * scale
            pane_cols = int(maps.shape[-1]) * scale
        strip = torch.zeros(
            (pane_rows, pane_count * pane_cols, 3),
            device=maps.device,
            dtype=torch.uint8,
        )
    else:
        strip = tacmap_strip(
            maps,
            tacmap_raw=maps,
            surface_points=surface_points,
            surface_valid=surface_valid,
            scale=max(1, int(args_cli.tacmap_scale)),
            gamma=float(args_cli.tacmap_gamma),
            view="raw",
            display_max_m=display_max_m,
            target_size=target_size,
            resize_mode=str(args_cli.tacmap_resize_mode),
            joint_marker_depth_values=marker_depth,
            joint_marker_depth_valid=marker_depth_valid,
            joint_source_indices=joint_source_indices,
            joint_source_weights=joint_source_weights,
            joint_pixel_valid=joint_pixel_valid,
        )
    strip = overlay_visual_local_tacmap_refinement(
        strip,
        local_refinement,
        pane_index=local_refinement_pane,
        pane_count=pane_count,
        display_max_m=display_max_m,
    )
    strip_cpu = to_numpy_uint8_rgb(strip)
    pane_width = max(1, int(strip_cpu.shape[1]) // pane_count)
    panes = [strip_cpu[:, index * pane_width : (index + 1) * pane_width] for index in range(pane_count)]
    return tile_rgb(panes, cols=pane_count)


def build_image(
    env,
    pressure_taxel_cache: list[tuple[str, np.ndarray, np.ndarray]],
    pressure_display_diffusion_kernels: dict[str, np.ndarray] | None = None,
    taxim_image: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if fots_baseline_is_enabled(env):
        return build_fots_baseline_image(env)
    if hydroshear_baseline_is_enabled(env):
        return build_hydroshear_baseline_image(env)
    if tacsl_baseline_is_enabled(env):
        return build_tacsl_baseline_image(env)

    pressure_maps = [force_map[0].reshape(-1) for force_map in pressure_maps_from_env(env)]
    pressure_counts = [int(force_map.numel()) for force_map in pressure_maps]
    if pressure_maps:
        pressure_flat_cpu = torch.cat(pressure_maps, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
        pressure_values = np.split(pressure_flat_cpu, np.cumsum(pressure_counts[:-1]))
    else:
        pressure_flat_cpu = np.zeros((0,), dtype=np.float32)
        pressure_values = []
    pressure_from_policy = getattr(env, "_rl_pressure_visual_source", "") == "policy_observation"
    pressure_panes = [
        pressure_layout_pane(
            force_values,
            points_l,
            diffusion_kernel=(
                None
                if pressure_from_policy or pressure_display_diffusion_kernels is None
                else pressure_display_diffusion_kernels.get(str(link_name))
            ),
            diffusion_blend=float(args_cli.pressure_display_diffusion_blend),
            spacing_px=max(1, int(args_cli.pressure_scale)),
            gamma=args_cli.pressure_gamma,
        )
        for force_values, (link_name, points_l, _normals_l) in zip(
            pressure_values, pressure_taxel_cache, strict=True
        )
    ]
    pressure_img = tile_rgb(pressure_panes, cols=6, preserve_scale=True)
    focus_pressure_img = None
    if bool(args_cli.whole_hand_pressure_only):
        image = pressure_img
    elif bool(args_cli.focus_pressure_only):
        focus_pressure_link = pressure_pad_link_name(args_cli.focus_finger, args_cli.pressure_pad_segment)
        pressure_links = [str(link_name) for link_name, _points_l, _normals_l in pressure_taxel_cache]
        if focus_pressure_link not in pressure_links:
            raise RuntimeError(
                f"Focus pressure pad {focus_pressure_link!r} is absent from the visualization cache: {pressure_links}"
            )
        focus_pressure_img = pressure_panes[pressure_links.index(focus_pressure_link)]

    tacmap_maps = tacmap_maps_from_env(env)[0]
    tacmap_display_maps = display_tacmap_maps(tacmap_maps)
    tacmap_target_size = tacmap_display_target_size()
    target_rows = (
        int(tacmap_target_size[0])
        if tacmap_target_size is not None
        else int(tacmap_display_maps.shape[-2]) * max(1, int(args_cli.tacmap_scale))
    )
    target_cols = (
        int(tacmap_target_size[1])
        if tacmap_target_size is not None
        else int(tacmap_display_maps.shape[-1]) * max(1, int(args_cli.tacmap_scale))
    )
    local_refinement = getattr(env, "_rl_visual_local_tacmap_refinement", None)
    if isinstance(local_refinement, dict):
        # Strict data ownership for this mode:
        # 32x24 selects the ROI, local rays produce the depth image, and
        # calibrated Marker rays remain exclusive to HydroShear.
        marker_depth = marker_depth_valid = None
        joint_indices = joint_weights = joint_pixel_valid = None
        surface_points = surface_valid = None
    else:
        marker_depth, marker_depth_valid, joint_indices, joint_weights, joint_pixel_valid = (
            tacmap_marker_joint_inputs(
                env,
                source_rows=int(tacmap_display_maps.shape[-2]),
                source_cols=int(tacmap_display_maps.shape[-1]),
                target_rows=target_rows,
                target_cols=target_cols,
            )
        )
        surface_points, surface_valid = tacmap_display_surface_geometry(env)
    tacmap_display_max_m = float(args_cli.tacmap_display_max_mm) * 1.0e-3
    if args_cli.show_all_tacmap_fingers:
        tacmap_img = batched_tacmap_image(
            tacmap_display_maps,
            surface_points=surface_points,
            surface_valid=surface_valid,
            marker_depth=marker_depth,
            marker_depth_valid=marker_depth_valid,
            joint_source_indices=joint_indices,
            joint_source_weights=joint_weights,
            joint_pixel_valid=joint_pixel_valid,
            display_max_m=tacmap_display_max_m,
            target_size=tacmap_target_size,
            local_refinement=local_refinement,
            local_refinement_pane=FINGER_CHOICES.index(str(args_cli.focus_finger)),
        )
    else:
        idx = FINGER_CHOICES.index(args_cli.focus_finger)
        tacmap_img = batched_tacmap_image(
            tacmap_display_maps[idx : idx + 1],
            surface_points=None if surface_points is None else surface_points[idx : idx + 1],
            surface_valid=None if surface_valid is None else surface_valid[idx : idx + 1],
            marker_depth=None if marker_depth is None else marker_depth[idx : idx + 1],
            marker_depth_valid=None if marker_depth_valid is None else marker_depth_valid[idx : idx + 1],
            joint_source_indices=None if joint_indices is None else joint_indices[idx : idx + 1],
            joint_source_weights=None if joint_weights is None else joint_weights[idx : idx + 1],
            joint_pixel_valid=None if joint_pixel_valid is None else joint_pixel_valid[idx : idx + 1],
            display_max_m=tacmap_display_max_m,
            target_size=tacmap_target_size,
            local_refinement=local_refinement,
            local_refinement_pane=0,
        )

    hydroshear_img = build_hydroshear_marker_image(env)
    if bool(args_cli.focus_pressure_only):
        assert focus_pressure_img is not None
        image = focus_pressure_img
    elif bool(args_cli.focus_visuotactile_only):
        # Keep the three focus-finger camera-aligned panes at native scale and
        # in semantic order: Depth, RGB, Marker.
        observation_panes = [tacmap_img]
        if taxim_image is not None:
            observation_panes.append(taxim_image)
        if hydroshear_img is not None:
            observation_panes.append(hydroshear_img)
        image = tile_rgb(observation_panes, cols=len(observation_panes), gap=8, preserve_scale=True)
    else:
        observation_panes = [pressure_img, tacmap_img]
        if hydroshear_img is not None:
            observation_panes.append(hydroshear_img)
        image = tile_rgb(observation_panes, cols=len(observation_panes), gap=8, preserve_scale=True)
        if taxim_image is not None:
            # Preserve the original full-panel layout outside the compact mode.
            image = tile_rgb([image, taxim_image], cols=1, gap=8, preserve_scale=True)

    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    focus_tacmap = tacmap_maps[focus_idx] if int(tacmap_maps.shape[0]) > focus_idx else tacmap_maps.reshape(-1)
    displayed_tacmap = (
        local_refinement.get("depth_m")
        if isinstance(local_refinement, dict)
        and isinstance(local_refinement.get("depth_m"), torch.Tensor)
        else focus_tacmap
    )
    pressure_max = float(np.max(pressure_flat_cpu)) if pressure_flat_cpu.size else 0.0
    axis_disp = getattr(env, "_rl_dynamic_axis_disp_m", None)
    axis_value = (
        axis_disp[0].detach().to(dtype=torch.float32) * 1000.0
        if isinstance(axis_disp, torch.Tensor) and axis_disp.numel()
        else torch.zeros((), device=env.device, dtype=torch.float32)
    )
    summary_gpu = torch.stack(
        (
            torch.amax(displayed_tacmap).to(dtype=torch.float32) * 1000.0,
            torch.count_nonzero(displayed_tacmap > 0.0).to(dtype=torch.float32),
            axis_value,
        )
    )
    tacmap_max_mm, tacmap_active_float, axis_disp_mm = summary_gpu.detach().cpu().tolist()
    tacmap_active = int(tacmap_active_float)
    tacmap_count = int(displayed_tacmap.numel())
    env._rl_visual_focus_tacmap_active = tacmap_active > 0
    applied_force_n = float(getattr(env, "_rl_dynamic_axis_force_n", 0.0))
    contact_force_n = float(getattr(env, "_rl_focus_contact_force_n", 0.0))
    normal_force_n = float(getattr(env, "_rl_focus_normal_force_n", 0.0))
    tacmap_link = str(getattr(env, "_rl_tactile_viz_tacmap_link", "unknown"))
    pressure_link = str(getattr(env, "_rl_tactile_viz_pressure_link", "unknown"))
    display_ops = (
        f"rot90={int(args_cli.tacmap_display_rot90) % 4},"
        f"flip_x={int(bool(args_cli.tacmap_display_flip_x))},"
        f"flip_y={int(bool(args_cli.tacmap_display_flip_y))}"
    )
    hydro_stats = hydroshear_marker_stats(env)
    if pressure_from_policy:
        pressure_display_stats = "pressure_display=exact_policy_obs"
    elif pressure_display_diffusion_kernels is not None:
        pressure_display_stats = (
            "pressure_display=gaussian_cpu "
            f"sigma={float(args_cli.pressure_display_diffusion_sigma_mm):g}mm "
            f"blend={float(args_cli.pressure_display_diffusion_blend):g}"
        )
    else:
        pressure_display_stats = "pressure_display=sensor_cache_fallback"
    pressure_rl_stats = (
        "RL_pressure=gaussian_gpu "
        f"sigma={float(TIANJI_RL_PRESSURE_DIFFUSION_SIGMA_M) * 1000.0:g}mm "
        f"blend={float(TIANJI_RL_PRESSURE_DIFFUSION_BLEND):g}"
        if bool(TIANJI_RL_PRESSURE_DIFFUSION_ENABLED)
        else "RL_pressure=raw"
    )
    marker_cache_ready = isinstance(
        getattr(env, "_rl_hydroshear_marker_depth_m", None),
        torch.Tensor,
    ) and bool(args_cli.tacmap_marker_depth_fusion)
    marker_fusion_stats = (
        f"depth_source=yellow_plane_from_local_{VISUAL_LOCAL_TACMAP_RAY_COUNT}_direct_integer_pixels "
        "marker_use=hydroshear_only roi_source=rl_32x24"
        if isinstance(local_refinement, dict)
        else "tacmap_fusion=joint_triangles(base+marker)"
        if marker_depth is not None and marker_cache_ready
        else "tacmap_fusion=waiting"
        if bool(args_cli.tacmap_marker_depth_fusion)
        else "tacmap_fusion=off"
    )
    local_bounds = local_refinement.get("display_bounds") if isinstance(local_refinement, dict) else None
    local_refinement_stats = (
        f"local_refine={VISUAL_LOCAL_TACMAP_RAY_COUNT} "
        f"roi_xy_mm=[{float(local_bounds[0]) * 1000.0:.2f}:{float(local_bounds[1]) * 1000.0:.2f},"
        f"{float(local_bounds[2]) * 1000.0:.2f}:{float(local_bounds[3]) * 1000.0:.2f}]"
        if local_bounds is not None
        else "local_refine=off"
    )
    policy_dense_depth = getattr(env, "_rl_ours_dense_tacmap_depth_m", None)
    tacmap_observation_stats = (
        f"tacmap_resnet_input={int(policy_dense_depth.shape[1])}x"
        f"{int(policy_dense_depth.shape[2])}x{int(policy_dense_depth.shape[3])}_metric_depth "
        "tacmap_rl_obs=128"
        if isinstance(policy_dense_depth, torch.Tensor) and policy_dense_depth.ndim == 4
        else "tacmap_resnet_input=waiting_for_5x240x320_metric_depth tacmap_rl_obs=128"
    )
    stats = (
        f"target={getattr(env, '_rl_tactile_viz_phase', 'unknown')} "
        f"pressure_max={pressure_max:.4f} tacmap_max={tacmap_max_mm:.3f}mm "
        f"tacmap_active={tacmap_active}/{tacmap_count} "
        f"applied={applied_force_n:.3f}N contact={contact_force_n:.3f}N "
        f"normal={normal_force_n:.3f}N axis_disp={axis_disp_mm:.3f}mm "
        f"focus={args_cli.focus_finger} pressure_pad={args_cli.pressure_pad_segment} "
        f"tacmap_link={tacmap_link} pressure_link={pressure_link} tacmap_display={display_ops} "
        f"{pressure_display_stats} {pressure_rl_stats} "
        f"{marker_fusion_stats} {local_refinement_stats} "
        f"pressure_obs={sum(pressure_counts)}[{','.join(str(count) for count in pressure_counts)}] "
        f"{tacmap_observation_stats} "
        f"{hydro_stats}"
    )
    return image, stats


def build_tacmap_contact_image(env) -> np.ndarray:
    tacmap_maps = tacmap_maps_from_env(env)[0]
    tacmap_display_maps = display_tacmap_maps(tacmap_maps)
    tacmap_display_max_m = float(args_cli.tacmap_display_max_mm) * 1.0e-3
    tacmap_target_size = tacmap_display_target_size()
    target_rows = (
        int(tacmap_target_size[0])
        if tacmap_target_size is not None
        else int(tacmap_display_maps.shape[-2]) * max(1, int(args_cli.tacmap_scale))
    )
    target_cols = (
        int(tacmap_target_size[1])
        if tacmap_target_size is not None
        else int(tacmap_display_maps.shape[-1]) * max(1, int(args_cli.tacmap_scale))
    )
    local_refinement = getattr(env, "_rl_visual_local_tacmap_refinement", None)
    if isinstance(local_refinement, dict):
        if args_cli.show_all_tacmap_fingers:
            selected_maps = tacmap_display_maps
            local_pane = FINGER_CHOICES.index(str(args_cli.focus_finger))
        else:
            focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
            selected_maps = tacmap_display_maps[focus_index : focus_index + 1]
            local_pane = 0
        return batched_tacmap_image(
            selected_maps,
            surface_points=None,
            surface_valid=None,
            marker_depth=None,
            marker_depth_valid=None,
            joint_source_indices=None,
            joint_source_weights=None,
            joint_pixel_valid=None,
            display_max_m=tacmap_display_max_m,
            target_size=tacmap_target_size,
            local_refinement=local_refinement,
            local_refinement_pane=local_pane,
        )
    marker_depth, marker_depth_valid, joint_indices, joint_weights, joint_pixel_valid = tacmap_marker_joint_inputs(
        env,
        source_rows=int(tacmap_display_maps.shape[-2]),
        source_cols=int(tacmap_display_maps.shape[-1]),
        target_rows=target_rows,
        target_cols=target_cols,
    )
    surface_points, surface_valid = tacmap_display_surface_geometry(env)
    if args_cli.show_all_tacmap_fingers:
        tacmap_panes = [
            tacmap_strip(
                tacmap_display_maps[i : i + 1],
                tacmap_raw=tacmap_display_maps[i : i + 1],
                surface_points=None if surface_points is None else surface_points[i : i + 1],
                surface_valid=None if surface_valid is None else surface_valid[i : i + 1],
                scale=max(1, int(args_cli.tacmap_scale)),
                gamma=float(args_cli.tacmap_gamma),
                view="raw",
                display_max_m=tacmap_display_max_m,
                target_size=tacmap_target_size,
                resize_mode=str(args_cli.tacmap_resize_mode),
                joint_marker_depth_values=None if marker_depth is None else marker_depth[i : i + 1],
                joint_marker_depth_valid=None if marker_depth_valid is None else marker_depth_valid[i : i + 1],
                joint_source_indices=None if joint_indices is None else joint_indices[i : i + 1],
                joint_source_weights=None if joint_weights is None else joint_weights[i : i + 1],
                joint_pixel_valid=None if joint_pixel_valid is None else joint_pixel_valid[i : i + 1],
            )
            for i in range(int(tacmap_maps.shape[0]))
        ]
        return tile_rgb(tacmap_panes, cols=5)

    idx = FINGER_CHOICES.index(args_cli.focus_finger)
    focus_pane = tacmap_strip(
        tacmap_display_maps[idx : idx + 1],
        tacmap_raw=tacmap_display_maps[idx : idx + 1],
        surface_points=None if surface_points is None else surface_points[idx : idx + 1],
        surface_valid=None if surface_valid is None else surface_valid[idx : idx + 1],
        scale=max(1, int(args_cli.tacmap_scale)),
        gamma=float(args_cli.tacmap_gamma),
        view="raw",
        display_max_m=tacmap_display_max_m,
        target_size=tacmap_target_size,
        resize_mode=str(args_cli.tacmap_resize_mode),
        joint_marker_depth_values=None if marker_depth is None else marker_depth[idx : idx + 1],
        joint_marker_depth_valid=None if marker_depth_valid is None else marker_depth_valid[idx : idx + 1],
        joint_source_indices=None if joint_indices is None else joint_indices[idx : idx + 1],
        joint_source_weights=None if joint_weights is None else joint_weights[idx : idx + 1],
        joint_pixel_valid=None if joint_pixel_valid is None else joint_pixel_valid[idx : idx + 1],
    )
    blank_pane = torch.zeros_like(focus_pane)
    tacmap_panes = [focus_pane if i == idx else blank_pane for i in range(len(FINGER_CHOICES))]
    return tile_rgb(tacmap_panes, cols=5)


def debug_tacmap_sensors(env, *, step: int) -> None:
    if not args_cli.debug_sensors:
        return
    if tacsl_baseline_is_enabled(env):
        debug_rows = int(TIANJI_TACSL_BASELINE_RAY_ROWS)
        debug_cols = int(TIANJI_TACSL_BASELINE_RAY_COLS)
    else:
        debug_rows = int(TIANJI_TACMAP_RL_ROWS)
        debug_cols = int(TIANJI_TACMAP_RL_COLS)
    obj = env.scene["object"]
    obj_pos = obj.data.root_pos_w[0].detach().cpu().numpy()
    obj_quat = obj.data.root_quat_w[0].detach().cpu().numpy()
    print(
        f"[debug step {step:06d}] object root pos={obj_pos.tolist()} quat={obj_quat.tolist()}",
        flush=True,
    )
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    pairs = (
        (TIANJI_TACMAP_OBJECT_SENSOR_NAMES[focus_idx], "object"),
        (TIANJI_TACMAP_SURFACE_SENSOR_NAMES[focus_idx], "surface"),
    )
    for sensor_name, kind in pairs:
        sensor = env.scene.sensors.get(str(sensor_name))
        if sensor is None:
            print(f"[debug step {step:06d}] {kind} sensor {sensor_name!r}: missing", flush=True)
            continue
        view_names = [type(view).__name__ for view in getattr(sensor, "_mesh_views", [])]
        target_exprs = [
            getattr(cfg, "prim_expr", "<missing>") for cfg in getattr(sensor, "_raycast_targets_cfg", [])
        ]
        raw = getattr(getattr(sensor, "data", None), "output", {}).get("distance_along_normal_raw")
        if isinstance(raw, torch.Tensor):
            raw0 = raw[0].reshape(-1)
            active = int(torch.count_nonzero(raw0 > 0.0).detach().cpu())
            max_m = float(torch.amax(raw0).detach().cpu()) if raw0.numel() else 0.0
            min_pos = (
                float(torch.amin(raw0[raw0 > 0.0]).detach().cpu())
                if bool(torch.any(raw0 > 0.0).detach().cpu())
                else 0.0
            )
        else:
            active = 0
            max_m = 0.0
            min_pos = 0.0
        starts_w = getattr(sensor, "_ray_starts_w", None)
        dirs_w = getattr(sensor, "_ray_directions_w", None)
        cop_x = cop_y = 0.0
        peak_x = peak_y = -1
        bbox_text = "none"
        if kind == "object":
            tacmap = getattr(env, "_rl_tacmap_penetration_m", None)
            if isinstance(tacmap, torch.Tensor) and tacmap.ndim == 4 and tacmap.shape[1] > focus_idx:
                depth = torch.nan_to_num(tacmap[0, focus_idx])
                weights = torch.clamp(depth, min=0.0)
                total = torch.sum(weights)
                if bool((total > 1.0e-12).detach().cpu()):
                    yy = torch.arange(depth.shape[0], dtype=torch.float32, device=depth.device).reshape(-1, 1)
                    xx = torch.arange(depth.shape[1], dtype=torch.float32, device=depth.device).reshape(1, -1)
                    cop_x = float((torch.sum(xx * weights) / total).detach().cpu())
                    cop_y = float((torch.sum(yy * weights) / total).detach().cpu())
                    peak = int(torch.argmax(depth.reshape(-1)).detach().cpu())
                    peak_y, peak_x = divmod(peak, int(depth.shape[1]))
                    active_px = torch.nonzero(depth > 0.0, as_tuple=False)
                    if active_px.numel():
                        y0 = int(torch.amin(active_px[:, 0]).detach().cpu())
                        y1 = int(torch.amax(active_px[:, 0]).detach().cpu())
                        x0 = int(torch.amin(active_px[:, 1]).detach().cpu())
                        x1 = int(torch.amax(active_px[:, 1]).detach().cpu())
                        bbox_text = f"({x0},{y0})-({x1},{y1})"
        if isinstance(starts_w, torch.Tensor) and isinstance(dirs_w, torch.Tensor) and starts_w.numel() > 0:
            center_i = (debug_rows // 2) * debug_cols + (debug_cols // 2)
            ray_start = starts_w[0, center_i].detach().cpu().numpy()
            ray_dir = dirs_w[0, center_i].detach().cpu().numpy()
            ray_dir = ray_dir / max(float(np.linalg.norm(ray_dir)), 1.0e-8)
            obj_delta = obj_pos - ray_start
            obj_along = float(np.dot(obj_delta, ray_dir))
            obj_perp = float(np.linalg.norm(obj_delta - obj_along * ray_dir))
            corners_l = presser_bbox_corners_l()
            if corners_l.size:
                corners_w = obj_pos.reshape(1, 3) + quat_rotate_np(obj_quat, corners_l)
                corner_delta = corners_w - ray_start.reshape(1, 3)
                corner_along = corner_delta @ ray_dir.reshape(3, 1)
                bbox_along_min = float(np.min(corner_along))
                bbox_along_max = float(np.max(corner_along))
            else:
                bbox_along_min = 0.0
                bbox_along_max = 0.0
            starts0 = starts_w[0].detach().reshape(-1, 3)
            dirs0 = dirs_w[0].detach().reshape(-1, 3)
            obj_pos_t = torch.as_tensor(obj_pos, device=starts0.device, dtype=starts0.dtype)
            delta = obj_pos_t.unsqueeze(0) - starts0
            dirs_norm = dirs0 / torch.clamp(torch.linalg.norm(dirs0, dim=-1, keepdim=True), min=1.0e-8)
            along = torch.sum(delta * dirs_norm, dim=-1, keepdim=True)
            tangent = delta - along * dirs_norm
            nearest_i = int(torch.argmin(torch.linalg.norm(tangent, dim=-1)).detach().cpu())
            obj_proj_y, obj_proj_x = divmod(nearest_i, debug_cols)
            obj_proj_perp = float(torch.linalg.norm(tangent[nearest_i]).detach().cpu())
        else:
            obj_along = 0.0
            obj_perp = 0.0
            bbox_along_min = 0.0
            bbox_along_max = 0.0
            obj_proj_x = obj_proj_y = -1
            obj_proj_perp = 0.0
        print(
            f"[debug step {step:06d}] {kind} sensor={sensor_name} "
            f"views={view_names} targets={target_exprs} active={active}/{debug_rows * debug_cols} "
            f"min_pos={min_pos * 1000.0:.3f}mm max={max_m * 1000.0:.3f}mm "
            f"obj_along_center_ray={obj_along * 1000.0:.3f}mm obj_perp_center_ray={obj_perp * 1000.0:.3f}mm "
            f"bbox_along=[{bbox_along_min * 1000.0:.3f},{bbox_along_max * 1000.0:.3f}]mm "
            f"tacmap_cop_px=({cop_x:.2f},{cop_y:.2f}) peak_px=({peak_x},{peak_y}) "
            f"active_bbox_px={bbox_text} object_proj_px=({obj_proj_x},{obj_proj_y}) "
            f"object_proj_perp={obj_proj_perp * 1000.0:.3f}mm",
            flush=True,
        )


def debug_pressure_sensors(env, *, step: int, pressure_link: str) -> None:
    if not args_cli.debug_sensors:
        return
    if tacsl_baseline_is_enabled(env):
        return
    try:
        sensor_idx = TIANJI_PRESSURE_PAD_LINK_ORDER.index(str(pressure_link))
    except ValueError:
        print(f"[debug step {step:06d}] pressure link {pressure_link!r}: not configured", flush=True)
        return
    sensor_name = str(TIANJI_PRESSURE_SENSOR_NAMES[sensor_idx])
    sensor = env.scene.sensors.get(sensor_name)
    if sensor is None:
        print(f"[debug step {step:06d}] pressure sensor {sensor_name!r}: missing", flush=True)
        return
    data = getattr(sensor, "data", None)
    if data is None:
        print(f"[debug step {step:06d}] pressure sensor={sensor_name}: missing data", flush=True)
        return

    def _stats(tensor: torch.Tensor | None) -> tuple[float, float, int, int]:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return 0.0, 0.0, 0, 0
        source = tensor.detach()
        if source.ndim == 4:
            source = source[0, 0]
        elif source.ndim == 3:
            source = source[0]
        source = torch.nan_to_num(source.reshape(-1))
        active = int(torch.count_nonzero(source > 0.0).detach().cpu())
        return (
            float(torch.amin(source).detach().cpu()),
            float(torch.amax(source).detach().cpu()),
            active,
            int(source.numel()),
        )

    sdf_min, sdf_max, sdf_positive, sdf_count = _stats(getattr(data, "signed_distance_map", None))
    pen_min, pen_max, pen_active, pen_count = _stats(getattr(data, "penetration_map", None))
    raw_min, raw_max, raw_active, raw_count = _stats(getattr(data, "pressure_force_map_raw", None))
    force_min, force_max, force_active, force_count = _stats(getattr(data, "pressure_force_map", None))

    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object"]
    _link_name, center_l_np, normal_l_np = pressure_pad_target(pressure_link)
    link_idx = body_index(robot, pressure_link)
    link_state = robot.data.body_link_state_w[0, link_idx, :7]
    link_pos = link_state[:3]
    link_quat = link_state[3:7]
    center_l = torch.as_tensor(center_l_np, device=env.device, dtype=torch.float32).unsqueeze(0)
    normal_l = torch.as_tensor(normal_l_np, device=env.device, dtype=torch.float32).unsqueeze(0)
    pressure_center_w = link_pos + quat_apply(link_quat.unsqueeze(0), center_l)[0]
    pressure_normal_w = quat_apply(link_quat.unsqueeze(0), normal_l)[0]
    pressure_normal_w = pressure_normal_w / torch.clamp(torch.linalg.norm(pressure_normal_w), min=1.0e-8)
    obj_pos = obj.data.root_pos_w[0]
    root_delta = obj_pos - pressure_center_w
    root_along = float(torch.dot(root_delta, pressure_normal_w).detach().cpu())
    root_perp = float(torch.linalg.norm(root_delta - root_along * pressure_normal_w).detach().cpu())
    internal_center_delta_mm = 0.0
    internal_center_perp_mm = 0.0
    actual_query_min = 0.0
    actual_query_max = 0.0
    actual_query_min_idx = -1
    actual_query_taxel_along = 0.0
    actual_query_taxel_perp = 0.0
    actual_query_ok = False
    pose_sensors = getattr(sensor, "_pose_sensors", None)
    points_local = getattr(sensor, "_points_local_per_sensor", None)
    query_sdf = getattr(sensor, "query_sdf_world", None)

    def query_sdf_env0(points_w: torch.Tensor) -> torch.Tensor:
        mesh_pos_w = getattr(sensor, "_mesh_pos_w", None)
        if isinstance(mesh_pos_w, torch.Tensor) and mesh_pos_w.ndim == 2 and mesh_pos_w.shape[0] == env.num_envs:
            points_w = points_w.unsqueeze(0).expand(env.num_envs, -1, -1)
        return query_sdf(points_w)[0].detach()

    if pose_sensors and isinstance(points_local, torch.Tensor):
        pose_sensor = pose_sensors[0]
        if getattr(pose_sensor, "is_initialized", False):
            pose_sensor.update(0.0, force_recompute=True)
            sensor_data = getattr(pose_sensor, "data", None)
            if sensor_data is not None and sensor_data.pos_w is not None and sensor_data.quat_w is not None:
                sensor_pos_w = sensor_data.pos_w[0, 0]
                sensor_quat_w = sensor_data.quat_w[0, 0]
                local_center = points_local[0].mean(dim=0)
                internal_center_w = sensor_pos_w + quat_apply(sensor_quat_w.unsqueeze(0), local_center.unsqueeze(0))[0]
                center_delta = internal_center_w - pressure_center_w
                internal_center_delta_mm = float(torch.linalg.norm(center_delta).detach().cpu()) * 1000.0
                internal_center_perp_mm = (
                    float(torch.linalg.norm(center_delta - torch.dot(center_delta, pressure_normal_w) * pressure_normal_w).detach().cpu())
                    * 1000.0
                )
                if callable(query_sdf):
                    taxel_points_l = points_local[0]
                    taxel_points_w = sensor_pos_w.unsqueeze(0) + quat_apply(
                        sensor_quat_w.unsqueeze(0).expand(taxel_points_l.shape[0], -1),
                        taxel_points_l,
                    )
                    try:
                        taxel_sdf = query_sdf_env0(taxel_points_w)
                        if taxel_sdf.numel():
                            actual_query_min_idx = int(torch.argmin(taxel_sdf).detach().cpu())
                            actual_query_min = float(taxel_sdf[actual_query_min_idx].detach().cpu())
                            actual_query_max = float(torch.amax(taxel_sdf).detach().cpu())
                            taxel_delta = taxel_points_w[actual_query_min_idx] - pressure_center_w
                            actual_query_taxel_along = float(torch.dot(taxel_delta, pressure_normal_w).detach().cpu())
                            actual_query_taxel_perp = float(
                                torch.linalg.norm(taxel_delta - actual_query_taxel_along * pressure_normal_w).detach().cpu()
                            )
                            actual_query_ok = True
                    except (RuntimeError, ValueError) as exc:
                        print(f"[debug step {step:06d}] pressure actual taxel query failed: {exc}", flush=True)
    grid_sdf_min = 0.0
    grid_sdf_center = 0.0
    grid_u_at_min = 0.0
    grid_v_at_min = 0.0
    if callable(query_sdf):
        normal_np = pressure_normal_w.detach().cpu().numpy().astype(np.float64)
        normal_np = normal_np / max(float(np.linalg.norm(normal_np)), 1.0e-12)
        ref_np = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if abs(float(np.dot(ref_np, normal_np))) > 0.95:
            ref_np = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        u_np = np.cross(ref_np, normal_np)
        u_np = u_np / max(float(np.linalg.norm(u_np)), 1.0e-12)
        v_np = np.cross(normal_np, u_np)
        v_np = v_np / max(float(np.linalg.norm(v_np)), 1.0e-12)
        offsets_np = np.linspace(-0.045, 0.045, 31, dtype=np.float32)
        uu_np, vv_np = np.meshgrid(offsets_np, offsets_np, indexing="xy")
        center_np = pressure_center_w.detach().cpu().numpy().astype(np.float64)
        points_np = (
            center_np.reshape(1, 3)
            + uu_np.reshape(-1, 1).astype(np.float64) * u_np.reshape(1, 3)
            + vv_np.reshape(-1, 1).astype(np.float64) * v_np.reshape(1, 3)
        )
        points_w = torch.as_tensor(points_np, device=env.device, dtype=torch.float32)
        try:
            sdf_grid = query_sdf_env0(points_w)
            if sdf_grid.numel():
                min_idx = int(torch.argmin(sdf_grid).detach().cpu())
                grid_sdf_min = float(sdf_grid[min_idx].detach().cpu())
                center_idx = int((len(offsets_np) * len(offsets_np)) // 2)
                grid_sdf_center = float(sdf_grid[center_idx].detach().cpu())
                grid_u_at_min = float(uu_np.reshape(-1)[min_idx])
                grid_v_at_min = float(vv_np.reshape(-1)[min_idx])
        except (RuntimeError, ValueError) as exc:
            print(f"[debug step {step:06d}] pressure sdf grid query failed: {exc}", flush=True)
    bbox_along_min = 0.0
    bbox_along_max = 0.0
    bbox_perp_min = 0.0
    bbox_perp_max = 0.0
    corners_l = presser_bbox_corners_l()
    if corners_l.size:
        obj_pos_np = obj_pos.detach().cpu().numpy()
        obj_quat_np = obj.data.root_quat_w[0].detach().cpu().numpy()
        pressure_center_np = pressure_center_w.detach().cpu().numpy()
        pressure_normal_np = pressure_normal_w.detach().cpu().numpy()
        corners_w = obj_pos_np.reshape(1, 3) + quat_rotate_np(obj_quat_np, corners_l)
        corner_delta = corners_w - pressure_center_np.reshape(1, 3)
        corner_along = corner_delta @ pressure_normal_np.reshape(3, 1)
        corner_tangent = corner_delta - corner_along * pressure_normal_np.reshape(1, 3)
        corner_perp = np.linalg.norm(corner_tangent, axis=1)
        bbox_along_min = float(np.min(corner_along))
        bbox_along_max = float(np.max(corner_along))
        bbox_perp_min = float(np.min(corner_perp))
        bbox_perp_max = float(np.max(corner_perp))

    print(
        f"[debug step {step:06d}] pressure sensor={sensor_name} link={pressure_link} "
        f"sdf=[{sdf_min * 1000.0:.3f},{sdf_max * 1000.0:.3f}]mm pos={sdf_positive}/{sdf_count} "
        f"penetration=[{pen_min * 1000.0:.3f},{pen_max * 1000.0:.3f}]mm active={pen_active}/{pen_count} "
        f"raw=[{raw_min:.4f},{raw_max:.4f}] active={raw_active}/{raw_count} "
        f"force=[{force_min:.4f},{force_max:.4f}] active={force_active}/{force_count} "
        f"root_along_normal={root_along * 1000.0:.3f}mm root_perp={root_perp * 1000.0:.3f}mm "
        f"internal_center_delta={internal_center_delta_mm:.3f}mm "
        f"internal_center_perp={internal_center_perp_mm:.3f}mm "
        f"bbox_along_normal=[{bbox_along_min * 1000.0:.3f},{bbox_along_max * 1000.0:.3f}]mm "
        f"bbox_perp=[{bbox_perp_min * 1000.0:.3f},{bbox_perp_max * 1000.0:.3f}]mm "
        f"actual_query=[{actual_query_min * 1000.0:.3f},{actual_query_max * 1000.0:.3f}]mm "
        f"taxel_min_idx={actual_query_min_idx} "
        f"taxel_min_along={actual_query_taxel_along * 1000.0:.3f}mm "
        f"taxel_min_perp={actual_query_taxel_perp * 1000.0:.3f}mm "
        f"actual_query_ok={int(actual_query_ok)} "
        f"sdf_grid_center={grid_sdf_center * 1000.0:.3f}mm "
        f"sdf_grid_min={grid_sdf_min * 1000.0:.3f}mm@"
        f"({grid_u_at_min * 1000.0:.1f},{grid_v_at_min * 1000.0:.1f})mm",
        flush=True,
    )


class PointNormalViz:
    def __init__(
        self,
        root_path: str,
        *,
        point_color: tuple[float, float, float],
        normal_color: tuple[float, float, float],
        point_radius: float,
        line_width: float,
        normal_length: float,
        show_points: bool = True,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, Vt
        import omni.usd

        self._gf = Gf
        self._vt = Vt
        self._normal_length = float(normal_length)
        self._show_points = bool(show_points)
        stage = omni.usd.get_context().get_stage()
        root = Sdf.Path(root_path)
        UsdGeom.Xform.Define(stage, root)

        point_path = root.AppendPath("Points")
        points = UsdGeom.Points.Define(stage, point_path)
        points.CreatePointsAttr()
        points.CreateWidthsAttr(Vt.FloatArray([float(point_radius) * 2.0]))
        self._point_color = tuple(float(v) for v in point_color)
        self._point_color_attr = UsdGeom.Gprim(points.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(point_color[0]), float(point_color[1]), float(point_color[2]))])
        )
        UsdGeom.Primvar(self._point_color_attr).SetInterpolation(UsdGeom.Tokens.vertex)
        self._points = points
        self._point_width = float(point_radius) * 2.0
        self._last_point_count: int | None = None
        self._last_point_colors: np.ndarray | None = None

        curve_path = root.AppendPath("Normals")
        curves = UsdGeom.BasisCurves.Define(stage, curve_path)
        curves.CreateTypeAttr("linear")
        curves.CreateBasisAttr("bezier")
        curves.CreateCurveVertexCountsAttr()
        curves.CreateWidthsAttr(Vt.FloatArray([float(line_width)]))
        curves.CreatePointsAttr()
        UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(normal_color[0]), float(normal_color[1]), float(normal_color[2]))])
        )
        self._normals = curves
        self._last_normal_count: int | None = None
        if not self._show_points:
            self._points.GetWidthsAttr().Set(Vt.FloatArray([]))
            self._points.GetPointsAttr().Set(Vt.Vec3fArray([]))
            self._point_color_attr.Set(Vt.Vec3fArray([]))

    def update(
        self,
        points_w: np.ndarray,
        normals_w: np.ndarray | None,
        *,
        valid: np.ndarray | None = None,
        point_colors: np.ndarray | None = None,
        point_max: int = 0,
        normal_stride: int = 1,
    ) -> None:
        Gf = self._gf
        Vt = self._vt
        points = np.asarray(points_w, dtype=np.float32).reshape(-1, 3)
        colors = None if point_colors is None else np.asarray(point_colors, dtype=np.float32).reshape(-1, 3)
        if valid is not None:
            mask = np.asarray(valid, dtype=bool).reshape(-1)
            if mask.size == points.shape[0]:
                points = points[mask]
                if colors is not None and colors.shape[0] == mask.shape[0]:
                    colors = colors[mask]
                if normals_w is not None:
                    normals_w = np.asarray(normals_w, dtype=np.float32).reshape(-1, 3)[mask]
        finite = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1.0e-8)
        points = points[finite]
        if colors is not None:
            if colors.shape[0] == finite.shape[0]:
                colors = colors[finite]
            else:
                colors = None
        if normals_w is not None:
            normals_w = np.asarray(normals_w, dtype=np.float32).reshape(-1, 3)
            if normals_w.shape[0] == finite.shape[0]:
                normals_w = normals_w[finite]
            else:
                normals_w = None
        max_points = int(point_max)
        if max_points > 0 and points.shape[0] > max_points:
            stride = int(math.ceil(points.shape[0] / max_points))
            points = points[::stride]
            if colors is not None:
                colors = colors[::stride]
            if normals_w is not None:
                normals_w = normals_w[::stride]

        if self._show_points:
            point_count = int(points.shape[0])
            if self._last_point_count != point_count:
                self._points.GetWidthsAttr().Set(Vt.FloatArray([float(self._point_width)] * point_count))
            self._points.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
            )
            if colors is None:
                colors = np.tile(np.asarray(self._point_color, dtype=np.float32).reshape(1, 3), (point_count, 1))
            colors = np.clip(np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            if self._last_point_colors is None or not np.array_equal(colors, self._last_point_colors):
                self._point_color_attr.Set(
                    Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors])
                )
                self._last_point_colors = colors.copy()
            self._last_point_count = point_count

        if normals_w is None or points.shape[0] == 0:
            if self._last_normal_count != 0:
                self._normals.GetCurveVertexCountsAttr().Set(Vt.IntArray([]))
                self._normals.GetPointsAttr().Set(Vt.Vec3fArray([]))
                self._last_normal_count = 0
            return

        stride = max(1, int(normal_stride))
        normal_points = points[::stride]
        normals = normals_w[::stride]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_normals = np.isfinite(normals).all(axis=1) & (norms.reshape(-1) > 1.0e-8)
        normal_points = normal_points[valid_normals]
        normals = normals[valid_normals] / np.clip(norms[valid_normals], 1.0e-8, None)
        ends = normal_points + normals * self._normal_length
        segments = np.stack((normal_points, ends), axis=1).reshape(-1, 3)
        normal_count = int(normal_points.shape[0])
        if self._last_normal_count != normal_count:
            self._normals.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * normal_count))
        self._normals.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in segments])
        )
        self._last_normal_count = normal_count


class LineSegmentViz:
    """Draw independent world-frame line segments as linear USD curves."""

    def __init__(
        self,
        root_path: str,
        *,
        color: tuple[float, float, float],
        line_width: float,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, Vt
        import omni.usd

        self._gf = Gf
        self._vt = Vt
        self._line_width = max(1.0e-7, float(line_width))
        stage = omni.usd.get_context().get_stage()
        root = Sdf.Path(root_path)
        UsdGeom.Xform.Define(stage, root)
        curves = UsdGeom.BasisCurves.Define(stage, root.AppendPath("Lines"))
        curves.CreateTypeAttr("linear")
        curves.CreateCurveVertexCountsAttr()
        curves.CreateWidthsAttr(Vt.FloatArray([self._line_width]))
        curves.CreatePointsAttr()
        UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        )
        self._curves = curves

    def update(self, segments_w: np.ndarray) -> None:
        Gf = self._gf
        Vt = self._vt
        segments = np.asarray(segments_w, dtype=np.float32).reshape(-1, 2, 3)
        segments = segments[np.isfinite(segments).all(axis=(1, 2))]
        count = int(segments.shape[0])
        self._curves.GetCurveVertexCountsAttr().Set(Vt.IntArray([2] * count))
        self._curves.GetWidthsAttr().Set(Vt.FloatArray([self._line_width]))
        self._curves.GetPointsAttr().Set(
            Vt.Vec3fArray(
                [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in segments.reshape(-1, 3)]
            )
        )


def ours_rl_component_terms(env_cfg) -> list:
    """Return the independent OURS terms, with legacy fallback."""

    proprio = env_cfg.observations.proprio
    terms = [
        term
        for name in (
            "rl_ours_pressure",
            "rl_ours_tacmap_policy",
            "rl_ours_hydroshear",
            "rl_ours_taxim_rgb",
        )
        if (term := getattr(proprio, name, None)) is not None
    ]
    legacy = getattr(proprio, "rl_ours", None)
    return terms if terms else ([legacy] if legacy is not None else [])


def ours_rl_component_params(env_cfg) -> dict:
    """Read the shared underlying OURS configuration from any enabled term."""

    terms = ours_rl_component_terms(env_cfg)
    if not terms:
        return {}
    params = dict(getattr(terms[0], "params", {}) or {})
    component_cfg = params.get("component_cfg")
    return dict(component_cfg) if isinstance(component_cfg, dict) else params


def update_ours_rl_component_params(env_cfg, updates: dict) -> None:
    """Apply debug-only overrides to every independently registered OURS term."""

    for obs_term in ours_rl_component_terms(env_cfg):
        params = dict(getattr(obs_term, "params", {}) or {})
        component_cfg = params.get("component_cfg")
        if isinstance(component_cfg, dict):
            updated_component_cfg = dict(component_cfg)
            updated_component_cfg.update(updates)
            params["component_cfg"] = updated_component_cfg
        else:
            params.update(updates)
        obs_term.params = params


def configure_tacmap_debug(env_cfg) -> None:
    shell = float(args_cli.tacmap_contact_shell)
    update_ours_rl_component_params(
        env_cfg,
        {
            "tacmap_contact_shell_m": max(0.0, shell),
            "local_tacmap_contact_threshold_m": max(
                0.0,
                float(args_cli.tacmap_local_contact_threshold_mm) * 1.0e-3,
            ),
            "local_tacmap_roi_margin_m": max(
                0.0,
                float(args_cli.tacmap_local_roi_margin_mm) * 1.0e-3,
            ),
            "local_tacmap_cache_debug_output": bool(visual_local_tacmap_enabled()),
        },
    )

    if not bool(args_cli.show_surface_debug):
        return
    if bool(args_cli.show_native_tacmap_debug):
        for sensor_name in TIANJI_TACMAP_SURFACE_SENSOR_NAMES:
            cfg = getattr(env_cfg.scene, str(sensor_name), None)
            if cfg is None:
                continue
            cfg.debug_viz_link_surfaces = True
            cfg.debug_viz_rays = True
            cfg.debug_viz_hits = False
            cfg.debug_viz_max_points = int(args_cli.surface_debug_point_max)
            cfg.debug_viz_point_radius = float(args_cli.surface_debug_point_radius)
            cfg.debug_viz_normal_stride = int(args_cli.surface_debug_normal_stride)
            cfg.debug_viz_normal_length = float(args_cli.surface_debug_normal_length)
            cfg.debug_viz_ray_width = float(args_cli.surface_debug_ray_width)
        for sensor_name in TIANJI_TACMAP_OBJECT_SENSOR_NAMES:
            cfg = getattr(env_cfg.scene, str(sensor_name), None)
            if cfg is None:
                continue
            # Object-surface hit points (the red dots) are intentionally not rendered.
            # cfg.debug_viz_hits = True
            cfg.debug_viz_hits = False
            cfg.debug_viz_hits_external_mask = False
            cfg.debug_viz_rays = True
            cfg.debug_viz_max_points = int(args_cli.surface_debug_point_max)
            cfg.debug_viz_point_radius = float(args_cli.surface_debug_point_radius)
            cfg.debug_viz_ray_width = float(args_cli.surface_debug_ray_width)


def configure_hydroshear_debug(env_cfg) -> None:
    update_ours_rl_component_params(
        env_cfg,
        {
            "hydroshear_cache_debug_output": bool(args_cli.show_hydroshear_marker),
            "hydroshear_render_debug_marker_images": False,
            "hydroshear_debug_sensor_index": FINGER_CHOICES.index(str(args_cli.focus_finger)),
            "hydroshear_cache_marker_depth_output": bool(args_cli.tacmap_marker_depth_fusion),
        },
    )

    baseline_term = getattr(env_cfg.observations.proprio, "rl_hydroshear_baseline", None)
    if baseline_term is not None:
        params = dict(getattr(baseline_term, "params", {}) or {})
        params["cache_debug_output"] = bool(args_cli.show_hydroshear_marker)
        baseline_term.params = params


def configure_taxim_rgb_debug(env_cfg) -> None:
    """Apply visualizer Taxim options to the same configuration used by the RL term."""

    background_mode = str(args_cli.taxim_rgb_background).lower()
    if background_mode == "marker":
        background_path = str(VITAI_MARKER_RGB_BACKGROUND)
    elif background_mode == "markerless":
        background_path = str(VITAI_MARKER_BACKGROUND)
    else:
        background_path = ""
    updates = {
        "taxim_rgb_depth_scale": float(args_cli.taxim_rgb_depth_scale),
        "taxim_rgb_with_shadow": bool(args_cli.taxim_rgb_with_shadow),
        "taxim_rgb_background_path": background_path,
    }
    if args_cli.taxim_rgb_device is not None:
        updates["taxim_rgb_device"] = str(args_cli.taxim_rgb_device)
    update_ours_rl_component_params(env_cfg, updates)


def configure_pressure_debug(env_cfg) -> None:
    update_ours_rl_component_params(
        env_cfg,
        {
            "cache_pressure_output": True,
            "pressure_output_attr_name": "_rl_pressure_observation",
        },
    )

    if not bool(args_cli.debug_sensors):
        return
    for sensor_name in TIANJI_PRESSURE_SENSOR_NAMES:
        cfg = getattr(env_cfg.scene, str(sensor_name), None)
        if cfg is not None:
            cfg.store_debug_fields = True


def configure_pressure_pad_collision_properties(env_cfg) -> None:
    event = getattr(env_cfg.events, "zz_revo3_pressure_pad_collision_properties", None)
    if event is None:
        return
    if args_cli.pressure_pad_contact_offset is None and args_cli.pressure_pad_rest_offset is None:
        return
    params = dict(getattr(event, "params", {}) or {})
    if args_cli.pressure_pad_contact_offset is not None:
        params["contact_offset"] = float(args_cli.pressure_pad_contact_offset)
    if args_cli.pressure_pad_rest_offset is not None:
        params["rest_offset"] = float(args_cli.pressure_pad_rest_offset)
    event.params = params


def update_tacmap_active_hit_masks(env) -> None:
    penetration = getattr(env, "_rl_tacmap_penetration_m", None)
    if not isinstance(penetration, torch.Tensor) or penetration.ndim < 4:
        return
    focus_idx = FINGER_CHOICES.index(args_cli.focus_finger)
    show_all = bool(args_cli.surface_debug_all_fingers)
    for i, sensor_name in enumerate(TIANJI_TACMAP_OBJECT_SENSOR_NAMES):
        if i >= penetration.shape[1]:
            break
        sensor = env.scene.sensors.get(str(sensor_name))
        update_fn = None if sensor is None else getattr(sensor, "update_hit_points_viz_mask", None)
        if not callable(update_fn):
            continue
        valid = penetration[0, i] > 0.0 if show_all or i == focus_idx else torch.zeros_like(penetration[0, i], dtype=torch.bool)
        update_fn(valid, env_id=0)


def focus_selected_surface_point_arrays(
    env,
    marker_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read the focus finger's selected RL surface hits in one GPU-to-CPU transfer."""

    points = getattr(env, "_rl_tacmap_surface_points_w", None)
    valid = getattr(env, "_rl_tacmap_surface_valid", None)
    if not (isinstance(points, torch.Tensor) and isinstance(valid, torch.Tensor)):
        return None
    finger_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
    if points.ndim != 5 or valid.ndim != 4 or points.shape[:-1] != valid.shape:
        return None
    if int(points.shape[0]) <= 0 or int(points.shape[1]) <= finger_index:
        return None
    points_focus = points[0, finger_index].to(dtype=torch.float32)
    valid_focus = valid[0, finger_index].to(dtype=torch.float32)
    marker = np.asarray(marker_mask, dtype=bool)
    if tuple(marker.shape) != tuple(points_focus.shape[:2]):
        return None
    packed = torch.cat(
        (points_focus, valid_focus[..., None]),
        dim=-1,
    ).detach().cpu().numpy()
    point_colors = np.tile(
        np.asarray((0.0, 0.85, 1.0), dtype=np.float32),
        (*marker.shape, 1),
    )
    point_colors[marker] = np.asarray((1.0, 0.35, 0.0), dtype=np.float32)
    return packed[..., :3], packed[..., 3] > 0.5, point_colors


def tacmap_surface_debug_arrays(env) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    points = getattr(env, "_rl_tacmap_surface_points_w", None)
    normals = getattr(env, "_rl_tacmap_surface_normals_w", None)
    valid = getattr(env, "_rl_tacmap_surface_valid", None)
    if not (isinstance(points, torch.Tensor) and isinstance(normals, torch.Tensor) and isinstance(valid, torch.Tensor)):
        return None
    points0 = points[0]
    normals0 = normals[0]
    valid0 = valid[0]
    penetration = getattr(env, "_rl_tacmap_penetration_m", None)
    active0 = None
    if isinstance(penetration, torch.Tensor) and penetration.ndim == 4:
        active0 = torch.nan_to_num(penetration[0]) > 0.0
    if not bool(args_cli.surface_debug_all_fingers):
        idx = FINGER_CHOICES.index(args_cli.focus_finger)
        points0 = points0[idx : idx + 1]
        normals0 = normals0[idx : idx + 1]
        valid0 = valid0[idx : idx + 1]
        if active0 is not None and active0.shape[0] > idx:
            active0 = active0[idx : idx + 1]
    if isinstance(active0, torch.Tensor) and active0.shape == valid0.shape:
        valid0 = torch.where(torch.any(active0), active0, valid0)
    packed = torch.cat(
        (
            points0.to(dtype=torch.float32),
            normals0.to(dtype=torch.float32),
            valid0[..., None].to(dtype=torch.float32),
        ),
        dim=-1,
    ).detach().cpu().numpy()
    return packed[..., 0:3], packed[..., 3:6], packed[..., 6] > 0.5


def hydroshear_marker_frame_debug_arrays(
    env,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    output = getattr(env, "_rl_hydroshear_output", None)
    debug_fields = (
        getattr(output, "debug_marker_points_w", None),
        getattr(output, "debug_marker_normals_w", None),
        getattr(output, "debug_marker_row_axes_w", None),
        getattr(output, "debug_marker_col_axes_w", None),
    )
    slot = FINGER_CHOICES.index(args_cli.focus_finger)
    if all(isinstance(field, list) and len(field) > slot for field in debug_fields):
        points_w, normals_w, row_axes_w, col_axes_w = (
            np.asarray(field[slot], dtype=np.float32).reshape(-1, 3) for field in debug_fields
        )
        if len(points_w) and len(points_w) == len(normals_w) == len(row_axes_w) == len(col_axes_w):
            return (
                points_w,
                normals_w,
                row_axes_w,
                col_axes_w,
                np.ones(len(points_w), dtype=bool),
            )

    points = getattr(env, "_rl_tacmap_surface_points_w", None)
    normals = getattr(env, "_rl_tacmap_surface_normals_w", None)
    valid = getattr(env, "_rl_tacmap_surface_valid", None)
    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    marker_uv_fn = None if adapter is None else getattr(adapter, "_scaled_marker_uv_t", None)
    tangent_axes_fn = None if adapter is None else getattr(adapter, "_batched_marker_tangent_axes", None)
    marker_normals_fn = None if adapter is None else getattr(adapter, "_batched_marker_normals", None)
    if not (
        isinstance(points, torch.Tensor)
        and isinstance(normals, torch.Tensor)
        and isinstance(valid, torch.Tensor)
        and callable(marker_uv_fn)
        and callable(tangent_axes_fn)
        and callable(marker_normals_fn)
    ):
        return None
    if points.ndim != 5 or normals.shape != points.shape or valid.shape != points.shape[:-1]:
        return None
    if points.shape[1] <= slot:
        return None

    points0 = points[0, slot : slot + 1].detach().to(device=env.device, dtype=torch.float32)
    normals0 = normals[0, slot : slot + 1].detach().to(device=env.device, dtype=torch.float32)
    valid0 = valid[0, slot : slot + 1].detach().to(device=env.device, dtype=torch.bool)
    height, width = int(points0.shape[1]), int(points0.shape[2])
    with torch.no_grad():
        marker_uv = marker_uv_fn(width, height)
        xs = torch.clamp(torch.round(marker_uv[:, 0]).to(torch.long), 0, width - 1)
        ys = torch.clamp(torch.round(marker_uv[:, 1]).to(torch.long), 0, height - 1)
        marker_points = points0[:, ys, xs]
        marker_valid = valid0[:, ys, xs] & torch.isfinite(marker_points).all(dim=-1)
        row_axes, col_axes = tangent_axes_fn(points0, valid0, xs, ys)
        marker_normals = marker_normals_fn(normals0, row_axes, col_axes, xs, ys)

    return (
        marker_points.detach().cpu().numpy(),
        marker_normals.detach().cpu().numpy(),
        row_axes.detach().cpu().numpy(),
        col_axes.detach().cpu().numpy(),
        marker_valid.detach().cpu().numpy(),
    )


def build_hydroshear_roi_wireframe_segments(
    surface_points_w: np.ndarray,
    surface_raw_m: np.ndarray,
    surface_valid: np.ndarray,
    ray_direction_w: np.ndarray,
    *,
    surface_margin_m: float,
    invalid_depth_ratio: float,
    grid_stride: int,
    boundary_padding_cells: float = 0.0,
) -> np.ndarray:
    """Reconstruct the padded HydroShear UV/depth ROI as world-frame segments."""

    points = np.asarray(surface_points_w, dtype=np.float32)
    raw = np.asarray(surface_raw_m, dtype=np.float32)
    valid = np.asarray(surface_valid, dtype=bool)
    ray = np.asarray(ray_direction_w, dtype=np.float32).reshape(3)
    if points.ndim != 3 or points.shape[-1] != 3 or raw.shape != points.shape[:2] or valid.shape != raw.shape:
        return np.zeros((0, 2, 3), dtype=np.float32)
    height, width = raw.shape
    if height <= 0 or width <= 0 or not np.isfinite(ray).all():
        return np.zeros((0, 2, 3), dtype=np.float32)
    ray_norm = float(np.linalg.norm(ray))
    if ray_norm <= 1.0e-9:
        return np.zeros((0, 2, 3), dtype=np.float32)
    ray = ray / ray_norm
    raw_valid = valid & np.isfinite(raw) & (raw > 0.0) & np.isfinite(points).all(axis=-1)
    if not np.any(raw_valid):
        return np.zeros((0, 2, 3), dtype=np.float32)

    ray_starts = points - raw[..., None] * ray.reshape(1, 1, 3)
    # The calibrated 32x24 layout is row-wise fitted to camera_range_aug4, so
    # it is not one affine rectangle. Draw the exact starts instead of
    # rebuilding an outdated rectangle from average row/column vectors.
    near = ray_starts.astype(np.float32, copy=True)

    padding_cells = float(boundary_padding_cells)
    if not np.isfinite(padding_cells):
        padding_cells = 0.0
    padding_cells = max(0.0, padding_cells)
    if padding_cells > 0.0:
        col_deltas = ray_starts[:, 1:, :] - ray_starts[:, :-1, :]
        col_delta_valid = raw_valid[:, 1:] & raw_valid[:, :-1]
        row_deltas = ray_starts[1:, :, :] - ray_starts[:-1, :, :]
        row_delta_valid = raw_valid[1:, :] & raw_valid[:-1, :]

        def mean_valid_delta(deltas: np.ndarray, delta_valid: np.ndarray) -> np.ndarray | None:
            finite = delta_valid & np.isfinite(deltas).all(axis=-1)
            if not np.any(finite):
                return None
            return np.mean(deltas[finite], axis=0).astype(np.float32)

        col_vec = mean_valid_delta(col_deltas, col_delta_valid)
        row_vec = mean_valid_delta(row_deltas, row_delta_valid)
        if col_vec is not None and row_vec is not None:
            col_planar = col_vec - ray * float(np.dot(col_vec, ray))
            col_norm = float(np.linalg.norm(col_planar))
            if col_norm > 1.0e-9:
                col_axis = col_planar / col_norm
                row_planar = row_vec - ray * float(np.dot(row_vec, ray))
                row_planar = row_planar - col_axis * float(np.dot(row_planar, col_axis))
                row_norm = float(np.linalg.norm(row_planar))
                if row_norm > 1.0e-9:
                    row_axis = row_planar / row_norm
                    if width > 1:
                        grid_u = np.sum(ray_starts * col_axis.reshape(1, 1, 3), axis=-1)
                        row_u_min = np.min(np.where(raw_valid, grid_u, np.inf), axis=1)
                        row_u_max = np.max(np.where(raw_valid, grid_u, -np.inf), axis=1)
                        row_cell_width = (row_u_max - row_u_min) / float(width - 1)
                        row_cell_width = np.where(np.isfinite(row_cell_width), row_cell_width, 0.0)
                        near[:, 0, :] -= (
                            padding_cells * row_cell_width[:, None] * col_axis.reshape(1, 3)
                        )
                        near[:, -1, :] += (
                            padding_cells * row_cell_width[:, None] * col_axis.reshape(1, 3)
                        )
                    if height > 1:
                        grid_v = np.sum(ray_starts * row_axis.reshape(1, 1, 3), axis=-1)
                        row_count = np.count_nonzero(raw_valid, axis=1)
                        row_v = np.sum(np.where(raw_valid, grid_v, 0.0), axis=1)
                        row_v = row_v / np.maximum(row_count, 1)
                        first_step = max(0.0, float(row_v[1] - row_v[0]))
                        last_step = max(0.0, float(row_v[-1] - row_v[-2]))
                        near[0, :, :] -= padding_cells * first_step * row_axis.reshape(1, 3)
                        near[-1, :, :] += padding_cells * last_step * row_axis.reshape(1, 3)

    max_surface_depth = float(np.max(raw[raw_valid]))
    fallback_depth = max_surface_depth * max(0.0, float(invalid_depth_ratio))
    depth_limit = np.where(
        raw_valid,
        raw + max(0.0, float(surface_margin_m)),
        fallback_depth,
    ).astype(np.float32)
    far = near + depth_limit[..., None] * ray.reshape(1, 1, 3)

    stride = max(1, int(grid_stride))
    row_ids = sorted(set(range(0, height, stride)) | {height - 1})
    col_ids = sorted(set(range(0, width, stride)) | {width - 1})
    segments: list[np.ndarray] = []
    for row in row_ids:
        for col in range(width - 1):
            segments.append(np.stack((near[row, col], near[row, col + 1]), axis=0))
            segments.append(np.stack((far[row, col], far[row, col + 1]), axis=0))
    for col in col_ids:
        for row in range(height - 1):
            segments.append(np.stack((near[row, col], near[row + 1, col]), axis=0))
            segments.append(np.stack((far[row, col], far[row + 1, col]), axis=0))
    for row in row_ids:
        for col in col_ids:
            segments.append(np.stack((near[row, col], far[row, col]), axis=0))
    if not segments:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3)


def hydroshear_roi_wireframe_arrays(env) -> np.ndarray | None:
    """Build env 0's focus-finger ROI wireframe from cached TacMap geometry."""

    surface_points = getattr(env, "_rl_tacmap_surface_points_w", None)
    surface_raw = getattr(env, "_rl_tacmap_surface_raw_m", None)
    surface_valid = getattr(env, "_rl_tacmap_surface_valid", None)
    ray_directions = getattr(env, "_rl_tacmap_ray_directions_w", None)
    adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
    cfg = None if adapter is None else getattr(adapter, "cfg", None)
    if not (
        isinstance(surface_points, torch.Tensor)
        and isinstance(surface_raw, torch.Tensor)
        and isinstance(surface_valid, torch.Tensor)
        and isinstance(ray_directions, torch.Tensor)
        and cfg is not None
        and surface_points.ndim == 5
        and surface_raw.shape == surface_points.shape[:-1]
        and surface_valid.shape == surface_raw.shape
        and ray_directions.ndim == 3
    ):
        return None
    sensor_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
    if (
        surface_points.shape[0] <= 0
        or surface_points.shape[1] <= sensor_index
        or ray_directions.shape[0] <= 0
        or ray_directions.shape[1] <= sensor_index
    ):
        return None
    return build_hydroshear_roi_wireframe_segments(
        surface_points[0, sensor_index].detach().cpu().numpy(),
        surface_raw[0, sensor_index].detach().cpu().numpy(),
        surface_valid[0, sensor_index].detach().cpu().numpy(),
        ray_directions[0, sensor_index].detach().cpu().numpy(),
        surface_margin_m=float(getattr(cfg, "object_sample_roi_surface_margin_m")),
        invalid_depth_ratio=float(getattr(cfg, "object_sample_roi_invalid_depth_ratio")),
        grid_stride=max(1, int(args_cli.hydroshear_roi_grid_stride)),
        boundary_padding_cells=float(
            getattr(cfg, "object_sample_roi_boundary_padding_cells", 0.0)
        ),
    )


def pressure_pad_taxel_local_cache() -> list[tuple[str, np.ndarray, np.ndarray]]:
    specs = load_pressure_pad_specs_from_urdf(TIANJI_PRESSUREPAD_URDF, require_files=False)
    by_link = {str(spec.link_name): spec for spec in specs}
    cache: list[tuple[str, np.ndarray, np.ndarray]] = []
    for link_name in TIANJI_PRESSURE_PAD_LINK_ORDER:
        spec = by_link.get(str(link_name))
        if spec is None:
            continue
        taxel_map = spec.to_taxel_map()
        cache.append(
            (
                str(link_name),
                np.asarray(taxel_map.points_l, dtype=np.float32),
                np.asarray(taxel_map.normals_l, dtype=np.float32),
            )
        )
    return cache


def surface_visual_points(points: np.ndarray, normals: np.ndarray, offset_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if points.shape != normals.shape:
        raise ValueError(f"Surface point/normal shape mismatch: {points.shape} vs {normals.shape}")
    normal_norm = np.linalg.norm(normals, axis=-1)
    valid = np.isfinite(normals).all(axis=-1) & (normal_norm > 1.0e-9)
    rendered = points.copy()
    rendered[valid] += normals[valid] / normal_norm[valid, None] * float(offset_m)
    return rendered


def calibrated_marker_local_cache(
    focus_finger: str | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        distortion_valid = np.asarray(layout["distortion_valid"], dtype=bool).reshape(-1)
        for finger, link_name in zip(FINGER_CHOICES, TIANJI_TACMAP_LINK_ORDER, strict=True):
            if focus_finger is not None and finger != focus_finger:
                continue
            points = np.asarray(layout[f"{finger}_points_link_m"], dtype=np.float32).reshape(-1, 3)
            normals = np.asarray(layout[f"{finger}_normals_link"], dtype=np.float32).reshape(-1, 3)
            methods = np.asarray(layout[f"{finger}_method"]).astype(str).reshape(-1)
            if not (len(points) == len(normals) == len(methods) == len(distortion_valid)):
                raise ValueError(f"Calibrated marker array length mismatch for {finger}")
            normal_norm = np.linalg.norm(normals, axis=-1)
            finite = (
                np.isfinite(points).all(axis=-1)
                & np.isfinite(normals).all(axis=-1)
                & (normal_norm > 1.0e-9)
            )
            exact = finite & distortion_valid & (methods == "ray_hit")
            colors = np.tile(np.asarray((0.0, 0.2, 1.0), dtype=np.float32), (np.count_nonzero(exact), 1))
            cache.append(
                (
                    str(link_name),
                    points[exact],
                    normals[exact] / normal_norm[exact, None],
                    colors,
                )
            )
    return cache


def calibrated_camera_frame_local_cache(
    focus_finger: str,
    *,
    axis_length_m: float,
) -> list[tuple[str, np.ndarray]]:
    """Build the calibrated camera X/Y/Z axes in the focus rubber-link frame."""

    finger = str(focus_finger)
    try:
        finger_index = FINGER_CHOICES.index(finger)
    except ValueError as exc:
        raise ValueError(f"Unknown camera-frame finger: {finger!r}") from exc
    link_name = str(TIANJI_TACMAP_LINK_ORDER[finger_index])
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        origin_l = np.asarray(layout[f"{finger}_camera_origin_link_m"], dtype=np.float32).reshape(3)
        rotation_link_camera = np.asarray(
            layout[f"{finger}_camera_rotation_link"],
            dtype=np.float32,
        ).reshape(3, 3)
    if not np.isfinite(origin_l).all() or not np.isfinite(rotation_link_camera).all():
        raise ValueError(f"Calibrated camera frame for {finger!r} contains non-finite values")

    # p_link = R_link_camera @ p_camera + origin_link. With row vectors,
    # the three camera basis vectors expressed in link coordinates are R.T rows.
    axes_l = rotation_link_camera.T.copy()
    axis_norm = np.linalg.norm(axes_l, axis=1, keepdims=True)
    if np.any(axis_norm <= 1.0e-9):
        raise ValueError(f"Calibrated camera frame for {finger!r} has a degenerate rotation")
    axes_l /= axis_norm
    starts_l = np.repeat(origin_l.reshape(1, 3), 3, axis=0)
    ends_l = starts_l + axes_l * float(axis_length_m)
    return [(link_name, np.stack((starts_l, ends_l), axis=1).astype(np.float32))]


def camera_visible_surface_local_cache(
    sample_count: int,
    surface_offset_m: float,
    *,
    focus_finger: str | None = None,
    boundary_point_count: int = 128,
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray]],
    list[tuple[str, np.ndarray]],
    list[tuple[str, np.ndarray]],
]:
    import xml.etree.ElementTree as ET

    from scripts.project_vitai_markers_to_mesh import (
        FINGERS,
        camera_visible_xy_hull,
        element_transform,
        load_intrinsics,
        load_surface_triangles,
        ray_triangle_nearest,
        sample_camera_visible_surface,
    )

    camera_matrix, distortion, image_size = load_intrinsics(VITAI_CAMERA_INTRINSICS)
    root = ET.parse(VITAI_RIGHT_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}
    cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    boundary_cache: list[tuple[str, np.ndarray]] = []
    camera_plane_boundary_cache: list[tuple[str, np.ndarray]] = []
    camera_plane_rectangle_cache: list[tuple[str, np.ndarray]] = []
    summaries: list[str] = []
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        stored_hulls = {
            finger: (
                np.asarray(
                    layout[f"{finger}_camera_visible_hull_xy_m"],
                    dtype=np.float64,
                ).reshape(-1, 2)
                if f"{finger}_camera_visible_hull_xy_m" in layout
                else None
            )
            for finger in FINGER_CHOICES
        }

    for finger_index, finger in enumerate(FINGER_CHOICES):
        if focus_finger is not None and finger != focus_finger:
            continue
        config = FINGERS[finger]
        camera_joint = joints[config["camera_joint"]]
        surface_joint = joints[config["surface_joint"]]
        camera_xyz_parent, camera_rotation_parent = element_transform(camera_joint)
        surface_xyz_parent, surface_rotation_parent = element_transform(surface_joint)
        camera_origin_surface = surface_rotation_parent.T @ (
            camera_xyz_parent - surface_xyz_parent
        )
        camera_rotation_surface = surface_rotation_parent.T @ camera_rotation_parent
        _, triangles, triangle_normals = load_surface_triangles(
            VITAI_RIGHT_URDF,
            links[config["surface_link"]],
        )
        hull_xy = stored_hulls[finger]
        if hull_xy is None:
            points, normals, visible = sample_camera_visible_surface(
                triangles,
                triangle_normals,
                camera_origin_surface=camera_origin_surface,
                camera_rotation_surface=camera_rotation_surface,
                camera_matrix=camera_matrix,
                distortion=distortion,
                image_size=image_size,
                sample_count=int(sample_count),
                seed=finger_index,
            )
            hull_xy = camera_visible_xy_hull(
                points,
                visible,
                camera_origin_surface=camera_origin_surface,
                camera_rotation_surface=camera_rotation_surface,
            ).astype(np.float64)
            colors = np.tile(
                np.asarray((0.35, 0.0, 0.3), dtype=np.float32),
                (points.shape[0], 1),
            )
            colors[visible] = np.asarray((0.0, 1.0, 0.15), dtype=np.float32)
            cache.append((str(config["surface_link"]), points, normals, colors))
            hull_source = f"sampled={int(np.count_nonzero(visible))}/{len(visible)}"
        else:
            hull_source = f"stored_hull_vertices={len(hull_xy)}"

        boundary_segment_count = 0
        camera_plane_segment_count = 0
        camera_rectangle_segment_count = 0
        camera_xy_summary = "camera_xy_mm=unavailable"
        camera_rectangle_summary = "dense_ray_rectangle=shared_white_calibration"
        if len(hull_xy) >= 3:
            closed_hull_xy = np.concatenate((hull_xy, hull_xy[:1]), axis=0)
            hull_edge_lengths = np.linalg.norm(np.diff(closed_hull_xy, axis=0), axis=1)
            nonzero_edges = hull_edge_lengths > 1.0e-9
            if np.count_nonzero(nonzero_edges) >= 3:
                hull_xy = hull_xy[nonzero_edges]
                closed_hull_xy = np.concatenate((hull_xy, hull_xy[:1]), axis=0)
                hull_edge_lengths = np.linalg.norm(np.diff(closed_hull_xy, axis=0), axis=1)
                hull_arclength = np.concatenate(([0.0], np.cumsum(hull_edge_lengths)))
                total_length = float(hull_arclength[-1])
                sample_total = max(3, int(boundary_point_count))
                sample_arclength = np.linspace(0.0, total_length, sample_total, endpoint=False)
                edge_ids = np.minimum(
                    np.searchsorted(hull_arclength, sample_arclength, side="right") - 1,
                    len(hull_edge_lengths) - 1,
                )
                edge_alpha = (
                    sample_arclength - hull_arclength[edge_ids]
                ) / np.maximum(hull_edge_lengths[edge_ids], 1.0e-12)
                boundary_xy_camera = (
                    closed_hull_xy[edge_ids] * (1.0 - edge_alpha[:, None])
                    + closed_hull_xy[edge_ids + 1] * edge_alpha[:, None]
                )

                # Orthogonally project the visible-surface boundary along the
                # calibrated camera Z axis onto the camera XY plane (Z=0).
                # These are metric camera coordinates, not perspective pixels.
                boundary_camera_plane = np.column_stack(
                    (
                        boundary_xy_camera,
                        np.zeros(sample_total, dtype=np.float64),
                    )
                )
                boundary_camera_plane_surface = (
                    boundary_camera_plane @ camera_rotation_surface.T
                    + camera_origin_surface.reshape(1, 3)
                )
                camera_plane_segments = np.stack(
                    (
                        boundary_camera_plane_surface,
                        np.roll(boundary_camera_plane_surface, -1, axis=0),
                    ),
                    axis=1,
                )
                camera_plane_segment_count = int(len(camera_plane_segments))
                camera_plane_boundary_cache.append(
                    (
                        str(config["surface_link"]),
                        camera_plane_segments.astype(np.float32),
                    )
                )
                xy_min_mm = np.min(boundary_xy_camera, axis=0) * 1000.0
                xy_max_mm = np.max(boundary_xy_camera, axis=0) * 1000.0
                camera_xy_summary = (
                    f"camera_xy_mm=x[{xy_min_mm[0]:.3f},{xy_max_mm[0]:.3f}] "
                    f"y[{xy_min_mm[1]:.3f},{xy_max_mm[1]:.3f}]"
                )

                # Reproject the uniformly sampled camera-XY hull along camera +Z.
                # Each resulting segment is short and anchored back on the rubber mesh,
                # instead of joining sparse random samples with long 3D chords.
                optical_axis_surface = camera_rotation_surface[:, 2]
                boundary_points = np.full((sample_total, 3), np.nan, dtype=np.float64)
                boundary_valid = np.zeros(sample_total, dtype=bool)
                for boundary_index, xy_camera in enumerate(boundary_xy_camera):
                    ray_origin_surface = camera_origin_surface + camera_rotation_surface @ np.asarray(
                        (xy_camera[0], xy_camera[1], 0.0),
                        dtype=np.float64,
                    )
                    first_hit = ray_triangle_nearest(
                        ray_origin_surface,
                        optical_axis_surface,
                        triangles,
                    )
                    if first_hit is None:
                        continue
                    triangle_id, ray_distance = first_hit
                    second_origin = ray_origin_surface + (float(ray_distance) + 1.0e-5) * optical_axis_surface
                    second_hit = ray_triangle_nearest(
                        second_origin,
                        optical_axis_surface,
                        triangles,
                    )
                    if second_hit is not None:
                        triangle_id, second_distance = second_hit
                        ray_distance = float(ray_distance) + 1.0e-5 + float(second_distance)
                    point_surface = ray_origin_surface + float(ray_distance) * optical_axis_surface
                    normal_surface = triangle_normals[int(triangle_id)].copy()
                    if float(np.dot(normal_surface, optical_axis_surface)) < 0.0:
                        normal_surface *= -1.0
                    normal_length = float(np.linalg.norm(normal_surface))
                    if normal_length <= 1.0e-9:
                        continue
                    boundary_points[boundary_index] = (
                        point_surface
                        + normal_surface / normal_length * float(surface_offset_m)
                    )
                    boundary_valid[boundary_index] = True

                next_valid = np.roll(boundary_valid, -1)
                boundary_segments = np.stack(
                    (boundary_points, np.roll(boundary_points, -1, axis=0)),
                    axis=1,
                )[boundary_valid & next_valid]
                if len(boundary_segments):
                    boundary_segment_count = int(len(boundary_segments))
                    boundary_cache.append(
                        (
                            str(config["surface_link"]),
                            boundary_segments.astype(np.float32),
                        )
                    )
        summaries.append(
            f"{finger}={hull_source} "
            f"surface_boundary_segments={boundary_segment_count} "
            f"camera_plane_segments={camera_plane_segment_count} "
            f"camera_rectangle_segments={camera_rectangle_segment_count} "
            f"{camera_xy_summary} {camera_rectangle_summary}"
        )

    print(
        "[INFO] Camera-visible rubber boundary samples "
        "(sampling only; surface points are not rendered): "
        + ", ".join(summaries),
        flush=True,
    )
    return cache, boundary_cache, camera_plane_boundary_cache, camera_plane_rectangle_cache


def clip_triangle_to_frustum_boundary_segments(
    triangle_camera: np.ndarray,
    clip_plane_normals: np.ndarray,
    *,
    boundary_plane_count: int,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Use the offline layout generator's exact camera-frustum clipper."""

    from scripts.project_vitai_markers_to_mesh import (
        clip_triangle_to_frustum_boundary_segments as shared_clipper,
    )

    return shared_clipper(
        triangle_camera,
        clip_plane_normals,
        boundary_plane_count=boundary_plane_count,
        tolerance=tolerance,
    )


def camera_frustum_surface_boundary_local_cache(
    *,
    focus_finger: str,
    surface_offset_m: float,
    print_summary: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Return the camera_range_aug4 frustum/rubber boundary for one finger."""

    import xml.etree.ElementTree as ET

    from scripts.project_vitai_markers_to_mesh import (
        FINGERS,
        camera_frustum_surface_boundary_segments,
        element_transform,
        load_intrinsics,
        load_surface_triangles,
    )

    finger = str(focus_finger)
    if finger not in FINGERS:
        raise ValueError(f"Unknown camera-visible finger: {finger!r}")
    config = FINGERS[finger]
    camera_matrix, _distortion, image_size = load_intrinsics(VITAI_CAMERA_INTRINSICS)
    root = ET.parse(VITAI_RIGHT_URDF).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    links = {link.attrib["name"]: link for link in root.findall("link")}
    camera_xyz_parent, camera_rotation_parent = element_transform(joints[config["camera_joint"]])
    surface_xyz_parent, surface_rotation_parent = element_transform(joints[config["surface_joint"]])
    camera_origin_surface = surface_rotation_parent.T @ (camera_xyz_parent - surface_xyz_parent)
    camera_rotation_surface = surface_rotation_parent.T @ camera_rotation_parent
    _, triangles, triangle_normals = load_surface_triangles(
        VITAI_RIGHT_URDF,
        links[config["surface_link"]],
    )

    segments_l = camera_frustum_surface_boundary_segments(
        triangles,
        triangle_normals,
        camera_origin_surface=camera_origin_surface,
        camera_rotation_surface=camera_rotation_surface,
        camera_matrix=camera_matrix,
        image_size=image_size,
        surface_offset_m=float(surface_offset_m),
    )
    width_px, height_px = (int(value) for value in image_size)
    if bool(print_summary):
        print(
            "[INFO] camera_range_aug4 visible rubber boundary: "
            f"finger={finger} segments={len(segments_l)} image={width_px}x{height_px}",
            flush=True,
        )
    return [(str(config["surface_link"]), segments_l)]


def camera_frustum_boundary_camera_xy_local_cache(
    surface_boundary_cache: list[tuple[str, np.ndarray]],
    *,
    focus_finger: str,
) -> list[tuple[str, np.ndarray]]:
    """Orthogonally map the rubber boundary onto the calibrated camera XY plane."""

    if not surface_boundary_cache:
        return []
    finger = str(focus_finger)
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        camera_origin_l = np.asarray(
            layout[f"{finger}_camera_origin_link_m"],
            dtype=np.float64,
        ).reshape(3)
        camera_rotation_l = np.asarray(
            layout[f"{finger}_camera_rotation_link"],
            dtype=np.float64,
        ).reshape(3, 3)

    projected_cache: list[tuple[str, np.ndarray]] = []
    camera_xy_arrays: list[np.ndarray] = []
    for link_name, segments_l in surface_boundary_cache:
        segments = np.asarray(segments_l, dtype=np.float64).reshape(-1, 2, 3)
        segments_camera = (segments - camera_origin_l.reshape(1, 1, 3)) @ camera_rotation_l
        valid = np.isfinite(segments_camera).all(axis=-1)
        valid_segments = np.all(valid, axis=1)
        if not np.any(valid_segments):
            continue
        segments_camera = segments_camera[valid_segments]
        projected_camera = segments_camera.copy()
        projected_camera[..., 2] = 0.0
        projected_l = projected_camera @ camera_rotation_l.T + camera_origin_l.reshape(1, 1, 3)
        projected_cache.append((str(link_name), projected_l.astype(np.float32)))
        camera_xy_arrays.append(projected_camera[..., :2].reshape(-1, 2))

    if camera_xy_arrays:
        camera_xy_m = np.concatenate(camera_xy_arrays, axis=0)
        camera_xy_min_mm = np.min(camera_xy_m, axis=0) * 1000.0
        camera_xy_max_mm = np.max(camera_xy_m, axis=0) * 1000.0
        print(
            "[INFO] camera_range_aug4 boundary orthogonally mapped to camera XY plane: "
            f"finger={finger} "
            f"x=[{camera_xy_min_mm[0]:.3f},{camera_xy_max_mm[0]:.3f}]mm "
            f"y=[{camera_xy_min_mm[1]:.3f},{camera_xy_max_mm[1]:.3f}]mm "
            "z=0.000mm",
            flush=True,
        )
    return projected_cache


def calibrated_camera_ray_rectangle_local_cache(
    *,
    focus_finger: str,
) -> list[tuple[str, np.ndarray]]:
    """Load the exact white rectangle shared by RL and visualization."""

    finger = str(focus_finger)
    camera_rectangles, rectangle_shape, rectangle_path = _load_camera_ray_rectangles(
        VITAI_MARKER_LAYOUT
    )
    rectangle_camera = np.asarray(camera_rectangles[finger], dtype=np.float64).reshape(4, 3)
    with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
        camera_origin_l = np.asarray(
            layout[f"{finger}_camera_origin_link_m"],
            dtype=np.float64,
        ).reshape(3)
        camera_rotation_l = np.asarray(
            layout[f"{finger}_camera_rotation_link"],
            dtype=np.float64,
        ).reshape(3, 3)
    rectangle_l = rectangle_camera @ camera_rotation_l.T + camera_origin_l.reshape(1, 3)
    rectangle_segments_l = np.stack(
        (rectangle_l, np.roll(rectangle_l, -1, axis=0)),
        axis=1,
    ).astype(np.float32)
    height_px, width_px = rectangle_shape
    print(
        "[INFO] Shared RL/visual white camera-ray rectangle: "
        f"finger={finger} resolution={width_px}x{height_px} source={rectangle_path}",
        flush=True,
    )
    return [
        (
            str(TIANJI_TACMAP_LINK_ORDER[FINGER_CHOICES.index(finger)]),
            rectangle_segments_l,
        )
    ]


def calibrated_marker_world_arrays(
    env,
    cache: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    robot: Articulation = env.scene["robot"]
    points_all: list[torch.Tensor] = []
    normals_all: list[torch.Tensor] = []
    colors_all: list[np.ndarray] = []
    for link_name, points_l_np, normals_l_np, colors_np in cache:
        try:
            idx = body_index(robot, link_name)
        except RuntimeError:
            continue
        link_state = robot.data.body_link_state_w[0, idx, :7]
        points_l = torch.as_tensor(points_l_np, device=env.device, dtype=torch.float32)
        normals_l = torch.as_tensor(normals_l_np, device=env.device, dtype=torch.float32)
        quat = link_state[3:7].unsqueeze(0).expand(points_l.shape[0], -1)
        points_all.append(link_state[:3].unsqueeze(0) + quat_apply(quat, points_l))
        normals_w = quat_apply(quat, normals_l)
        normals_all.append(
            normals_w / torch.clamp(torch.linalg.norm(normals_w, dim=-1, keepdim=True), min=1.0e-8)
        )
        colors_all.append(colors_np)
    if not points_all:
        return None
    packed = torch.cat((torch.cat(points_all, dim=0), torch.cat(normals_all, dim=0)), dim=-1)
    packed_cpu = packed.detach().cpu().numpy()
    return packed_cpu[:, 0:3], packed_cpu[:, 3:6], np.concatenate(colors_all, axis=0)


def visual_local_tacmap_world_arrays(env) -> tuple[np.ndarray, np.ndarray] | None:
    """Return local samples colored only by penetration state."""

    state = getattr(env, "_rl_visual_local_tacmap_refinement", None)
    if not isinstance(state, dict):
        return None
    points_l = state.get("display_reference_points_l")
    valid = state.get("valid")
    depth_m = state.get("depth_m")
    link_name = state.get("link_name")
    if (
        not isinstance(points_l, torch.Tensor)
        or not isinstance(valid, torch.Tensor)
        or not isinstance(depth_m, torch.Tensor)
        or not isinstance(link_name, str)
    ):
        return None
    points_l = points_l.detach().reshape(-1, 3).to(device=env.device, dtype=torch.float32)
    valid_flat = valid.detach().reshape(-1).to(device=env.device, dtype=torch.bool)
    depth_flat = depth_m.detach().reshape(-1).to(device=env.device, dtype=torch.float32)
    count = min(
        int(points_l.shape[0]),
        int(valid_flat.numel()),
        int(depth_flat.numel()),
    )
    if count <= 0:
        return None
    points_l = points_l[:count]
    valid_flat = valid_flat[:count] & torch.isfinite(points_l).all(dim=-1)
    depth_flat = depth_flat[:count]

    robot: Articulation = env.scene["robot"]
    try:
        link_index = body_index(robot, link_name)
    except RuntimeError:
        return None
    link_state = robot.data.body_link_state_w[0, link_index, :7]
    quat = link_state[3:7].unsqueeze(0).expand(count, -1)
    points_w = link_state[:3].unsqueeze(0) + quat_apply(quat, points_l)
    no_penetration_color = torch.tensor((1.0, 0.85, 0.0), device=env.device, dtype=torch.float32)
    penetration_color = torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=torch.float32)
    colors = no_penetration_color.reshape(1, 3).expand(count, -1).clone()
    colors[depth_flat > 0.0] = penetration_color
    packed = torch.cat((points_w, colors, valid_flat.to(torch.float32).unsqueeze(-1)), dim=-1)
    packed_np = packed.detach().cpu().numpy()
    packed_np = packed_np[packed_np[:, 6] > 0.5]
    if len(packed_np) == 0:
        return None
    return packed_np[:, :3], packed_np[:, 3:6]


def visual_local_tacmap_ray_world_arrays(
    env,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the exact local object rays and the cached 320x240 reference boundary rays."""

    state = getattr(env, "_rl_visual_local_tacmap_refinement", None)
    if not isinstance(state, dict):
        return None
    reference_starts = state.get("reference_starts_l")
    reference_depth = state.get("reference_depth_m")
    sample_indices = state.get("sample_reference_pixel_indices")
    sample_reference_depth = state.get("sample_reference_depth_m")
    reference_boundary_indices = state.get("reference_boundary_indices")
    ray_direction = state.get("ray_direction_l")
    link_name = state.get("link_name")
    if not (
        isinstance(reference_starts, torch.Tensor)
        and isinstance(reference_depth, torch.Tensor)
        and isinstance(sample_indices, torch.Tensor)
        and isinstance(sample_reference_depth, torch.Tensor)
        and isinstance(reference_boundary_indices, torch.Tensor)
        and isinstance(ray_direction, torch.Tensor)
        and isinstance(link_name, str)
    ):
        return None

    write_count = int(state.get("sample_write_count", 0))
    write_count = min(
        write_count,
        int(sample_indices.numel()),
        int(sample_reference_depth.numel()),
    )
    if write_count <= 0:
        return None
    device = env.device
    flat_starts = reference_starts.detach().reshape(-1, 3).to(device=device, dtype=torch.float32)
    direction = ray_direction.detach().reshape(1, 3).to(device=device, dtype=torch.float32)
    direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1.0e-8)

    selected_indices = sample_indices[:write_count].to(device=device, dtype=torch.long)
    selected_starts = flat_starts.index_select(0, selected_indices)
    selected_reference_depth = sample_reference_depth[:write_count].detach().to(
        device=device,
        dtype=torch.float32,
    )
    # Display only the final 2 mm outside the calibrated rubber surface.  The
    # sensor still casts from the original camera-plane starts over its full
    # query distance; this clipping changes only the yellow USD debug lines.
    selected_surface_points = selected_starts + direction * selected_reference_depth.unsqueeze(-1)
    selected_display_starts = (
        selected_surface_points - direction * float(VISUAL_LOCAL_TACMAP_RAY_DISPLAY_LENGTH_M)
    )
    selected_segments_l = torch.stack((selected_display_starts, selected_surface_points), dim=1)

    reference_boundary_indices = reference_boundary_indices.detach().to(
        device=device,
        dtype=torch.long,
    ).reshape(-1)
    if int(reference_boundary_indices.numel()) > 0:
        reference_starts_l = flat_starts.index_select(0, reference_boundary_indices)
        reference_lengths = reference_depth.detach().reshape(-1).to(
            device=device,
            dtype=torch.float32,
        ).index_select(0, reference_boundary_indices)
        reference_ends_l = reference_starts_l + direction * reference_lengths.unsqueeze(-1)
        reference_segments_l = torch.stack((reference_starts_l, reference_ends_l), dim=1)
    else:
        reference_segments_l = torch.zeros((0, 2, 3), device=device, dtype=torch.float32)

    robot: Articulation = env.scene["robot"]
    try:
        link_index = body_index(robot, link_name)
    except RuntimeError:
        return None
    link_state = robot.data.body_link_state_w[0, link_index, :7]

    def segments_to_world(segments_l: torch.Tensor) -> np.ndarray:
        if int(segments_l.shape[0]) == 0:
            return np.zeros((0, 2, 3), dtype=np.float32)
        flat_points_l = segments_l.reshape(-1, 3)
        quat = link_state[3:7].unsqueeze(0).expand(flat_points_l.shape[0], -1)
        flat_points_w = link_state[:3].unsqueeze(0) + quat_apply(quat, flat_points_l)
        return flat_points_w.reshape(-1, 2, 3).detach().cpu().numpy()

    return segments_to_world(selected_segments_l), segments_to_world(reference_segments_l)


def local_line_segments_world(
    env,
    cache: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    robot: Articulation = env.scene["robot"]
    segments_all: list[torch.Tensor] = []
    for link_name, segments_l_np in cache:
        try:
            idx = body_index(robot, link_name)
        except RuntimeError:
            continue
        segments_l = torch.as_tensor(segments_l_np, device=env.device, dtype=torch.float32).reshape(-1, 3)
        link_state = robot.data.body_link_state_w[0, idx, :7]
        quat = link_state[3:7].unsqueeze(0).expand(segments_l.shape[0], -1)
        segments_all.append(link_state[:3].unsqueeze(0) + quat_apply(quat, segments_l))
    if not segments_all:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return torch.cat(segments_all, dim=0).reshape(-1, 2, 3).detach().cpu().numpy()


def pressure_pad_taxel_world_arrays(
    env,
    cache: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    visual_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # Color taxels from the exact post-Gaussian pressure slice used by the
    # policy observation. No pressure sensor is recomputed here.
    robot: Articulation = env.scene["robot"]
    pressure_by_link = {
        str(link_name): values[0].reshape(-1)
        for link_name, values in zip(TIANJI_PRESSURE_PAD_LINK_ORDER, pressure_maps_from_env(env), strict=True)
    }
    points_all: list[torch.Tensor] = []
    normals_all: list[torch.Tensor] = []
    colors_all: list[torch.Tensor] = []
    inactive_color = torch.tensor((1.0, 0.85, 0.0), device=env.device, dtype=torch.float32)
    active_color = torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=torch.float32)
    for link_name, points_l_np, normals_l_np in cache:
        try:
            idx = body_index(robot, link_name)
        except RuntimeError:
            continue
        link_state = robot.data.body_link_state_w[0, idx, :7]
        link_pos = link_state[:3]
        link_quat = link_state[3:7]
        points_l_np = surface_visual_points(points_l_np, normals_l_np, visual_offset_m)
        points_l = torch.as_tensor(points_l_np, device=env.device, dtype=torch.float32)
        normals_l = torch.as_tensor(normals_l_np, device=env.device, dtype=torch.float32)
        quat = link_quat.unsqueeze(0).expand(points_l.shape[0], -1)
        normals_w = quat_apply(quat, normals_l)
        normals_w = normals_w / torch.clamp(torch.linalg.norm(normals_w, dim=-1, keepdim=True), min=1.0e-8)
        points_w = link_pos.unsqueeze(0) + quat_apply(quat, points_l)
        points_all.append(points_w)
        normals_all.append(normals_w)

        active = torch.zeros((points_l.shape[0],), device=env.device, dtype=torch.bool)
        source = pressure_by_link.get(str(link_name))
        if isinstance(source, torch.Tensor) and source.numel() > 0:
            source = torch.nan_to_num(source.detach().to(device=env.device, dtype=torch.float32))
            if source.numel() == active.numel():
                active = source > 0.0
        colors = inactive_color.expand(points_l.shape[0], -1).clone()
        colors[active] = active_color
        colors_all.append(colors)
    if not points_all:
        return None
    points = torch.cat(points_all, dim=0)
    normals = torch.cat(normals_all, dim=0)
    colors = torch.cat(colors_all, dim=0)
    packed = torch.cat((points, normals, colors), dim=-1).detach().cpu().numpy()
    return packed[:, 0:3], packed[:, 3:6], packed[:, 6:9]


class ObsPanel:
    def __init__(self, image: np.ndarray, title: str = "RL Observation — Pressure / TacMap / HydroShear"):
        import omni.ui as ui

        h, w = image.shape[:2]
        self._height = int(h)
        self._width = int(w)
        self._win = ui.Window(str(title), width=self._width + 24, height=self._height + 112)
        with self._win.frame:
            with ui.VStack(spacing=6):
                ui.Label("LEFT: PRESSURE  |  CENTER: TACMAP  |  RIGHT: HYDROSHEAR (when available)")
                ui.Label("Pressure rows: mid M/P, index M/P, ring M/P  |  pinky M/P, thumb M/P, palm")
                self._stats = ui.Label("waiting")
                self._provider = ui.ByteImageProvider()
                ui.ImageWithProvider(self._provider, width=self._width, height=self._height)

    def update(self, image: np.ndarray, stats: str) -> None:
        rgb = to_numpy_uint8_rgb(image)
        alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=-1), dtype=np.uint8)
        self._provider.set_bytes_data(memoryview(rgba.reshape(-1)), (self._width, self._height))
        self._stats.text = stats


def maybe_show_cv(image: np.ndarray) -> None:
    if not args_cli.show_cv:
        return
    try:
        import cv2

        cv2.imshow("RL Tactile Observations", to_numpy_uint8_rgb(image)[:, :, ::-1])
        cv2.waitKey(1)
    except Exception as exc:
        print(f"[WARN] OpenCV display disabled: {exc}", flush=True)
        args_cli.show_cv = False


def pause_for_tacmap_onset(env, panel, image: np.ndarray, stats: str, step: int) -> None:
    print(
        f"[TACMAP_ONSET_PAUSE] step={step} focus={args_cli.focus_finger} {stats}",
        flush=True,
    )
    pause_fn = getattr(env.sim, "pause", None)
    if callable(pause_fn):
        pause_fn()
    if bool(getattr(args_cli, "headless", False)):
        return
    while simulation_app.is_running():
        if panel is not None:
            panel.update(image, f"PAUSED {stats}")
        maybe_show_cv(image)
        if env.sim.has_gui() or env.sim.has_rtx_sensors():
            env.sim.render()
        else:
            simulation_app.update()
        time.sleep(0.03)


def print_viewer_camera(step: int) -> None:
    if not bool(args_cli.print_viewer_camera):
        return
    every = max(1, int(args_cli.print_viewer_camera_every))
    if step % every != 0:
        return
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Usd, UsdGeom

        viewport = get_active_viewport()
        if viewport is None:
            return
        camera_path = str(viewport.camera_path)
        prim = viewport.stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            return
        mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = mat.ExtractTranslation()
        forward = mat.ExtractRotation().TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
        coi_attr = prim.GetAttribute("omni:kit:centerOfInterest")
        coi = coi_attr.Get() if coi_attr and coi_attr.IsValid() else None
        distance = max(0.1, float(Gf.Vec3d(coi).GetLength())) if coi is not None else 1.0
        target = eye + forward * distance
        print(
            "[VIEWER_CAMERA] "
            f"--viewer-eye {eye[0]:.6f} {eye[1]:.6f} {eye[2]:.6f} "
            f"--viewer-lookat {target[0]:.6f} {target[1]:.6f} {target[2]:.6f} "
            f"camera={camera_path}",
            flush=True,
        )
    except Exception as exc:
        print(f"[WARN] Failed to read active viewport camera: {exc}", flush=True)
        args_cli.print_viewer_camera = False


def configure_robot_asset(env_cfg) -> None:
    if args_cli.robot_asset == "right":
        return

    urdf_path = args_cli.bimanual_urdf.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Bimanual URDF not found: {urdf_path}")

    robot_cfg = env_cfg.scene.robot
    source_spawn = robot_cfg.spawn
    robot_cfg.spawn = sim_utils.UrdfFileCfg(
        asset_path=str(urdf_path),
        usd_dir=str(BIMANUAL_USD_DIR),
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type="convex_decomposition",
        self_collision=False,
        collision_from_visuals=False,
        articulation_props=getattr(source_spawn, "articulation_props", None),
        joint_drive_props=getattr(source_spawn, "joint_drive_props", None),
        rigid_props=getattr(source_spawn, "rigid_props", None),
        collision_props=getattr(source_spawn, "collision_props", None),
        activate_contact_sensors=getattr(source_spawn, "activate_contact_sensors", False),
    )

    init_state = robot_cfg.init_state
    joint_pos = dict(getattr(init_state, "joint_pos", {}) or {})
    joint_pos.update({"Joint[1-7]_L": 0.0, "left_.*_joint": 0.0, "Joint4_R": 1.0})
    init_state.pos = BIMANUAL_ROOT_POS
    init_state.rot = BIMANUAL_ROOT_ROT
    init_state.joint_pos = joint_pos

    actuators = dict(robot_cfg.actuators)
    actuators["tianji_left_arm_base"] = actuators["tianji_arm_base"].replace(joint_names_expr=["Joint[1-2]_L"])
    actuators["tianji_left_arm_mid"] = actuators["tianji_arm_mid"].replace(joint_names_expr=["Joint[3-4]_L"])
    actuators["tianji_left_arm_wrist"] = actuators["tianji_arm_wrist"].replace(joint_names_expr=["Joint[5-7]_L"])
    actuators["left_hand"] = actuators["brainco_hand"].replace(joint_names_expr=["left_.*_joint"])
    robot_cfg.actuators = actuators

    env_cfg.events.robot_physics_material = None
    for event_name in ("zz_revo3_touch_compliant_materials", "zz_revo3_pressure_pad_collision_properties"):
        getattr(env_cfg.events, event_name).mode = "prestartup"

    print(
        f"[INFO] Bimanual robot asset: urdf={urdf_path}, root_pos={BIMANUAL_ROOT_POS}, "
        f"root_rot={BIMANUAL_ROOT_ROT}, merge_fixed_joints=False",
        flush=True,
    )


def configure_env():
    device = getattr(args_cli, "device", None) or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=int(args_cli.num_envs))
    env_cfg.curriculum = None
    configure_robot_asset(env_cfg)
    dynamic_axis = dynamic_axis_enabled()
    if args_cli.presser != "task_object":
        collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=bool(args_cli.presser_collision_enabled),
        )
        if args_cli.presser_contact_offset is not None:
            collision_props.contact_offset = float(args_cli.presser_contact_offset)
        if args_cli.presser_rest_offset is not None:
            collision_props.rest_offset = float(args_cli.presser_rest_offset)
        env_cfg.scene.object.spawn = sim_utils.UsdFileCfg(
            usd_path=str(PRESSER_SPECS[args_cli.presser]["usd"]),
            scale=(float(args_cli.presser_scale),) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=not dynamic_axis,
                disable_gravity=True,
                enable_gyroscopic_forces=dynamic_axis,
                max_linear_velocity=(
                    float(args_cli.presser_axis_max_speed)
                    if dynamic_axis and float(args_cli.presser_axis_max_speed) > 0.0
                    else None
                ),
                max_depenetration_velocity=1000.0,
            ),
            collision_props=collision_props,
            mass_props=sim_utils.MassPropertiesCfg(mass=float(args_cli.presser_mass)),
        )
        env_cfg.scene.object.init_state.pos = (0.0, 0.0, 2.0)
        env_cfg.scene.object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    else:
        spawn_cfg = env_cfg.scene.object.spawn
        if hasattr(spawn_cfg, "rigid_props"):
            if spawn_cfg.rigid_props is None:
                spawn_cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg()
            spawn_cfg.rigid_props.rigid_body_enabled = True
            spawn_cfg.rigid_props.kinematic_enabled = not dynamic_axis
            spawn_cfg.rigid_props.disable_gravity = True
            spawn_cfg.rigid_props.enable_gyroscopic_forces = dynamic_axis
            spawn_cfg.rigid_props.max_linear_velocity = (
                float(args_cli.presser_axis_max_speed)
                if dynamic_axis and float(args_cli.presser_axis_max_speed) > 0.0
                else None
            )
            spawn_cfg.rigid_props.max_depenetration_velocity = 1000.0
        if hasattr(spawn_cfg, "collision_props"):
            if spawn_cfg.collision_props is None:
                spawn_cfg.collision_props = sim_utils.CollisionPropertiesCfg()
            spawn_cfg.collision_props.collision_enabled = bool(args_cli.presser_collision_enabled)
            if args_cli.presser_contact_offset is not None:
                spawn_cfg.collision_props.contact_offset = float(args_cli.presser_contact_offset)
            if args_cli.presser_rest_offset is not None:
                spawn_cfg.collision_props.rest_offset = float(args_cli.presser_rest_offset)
        if hasattr(spawn_cfg, "mass_props"):
            if spawn_cfg.mass_props is None:
                spawn_cfg.mass_props = sim_utils.MassPropertiesCfg()
            spawn_cfg.mass_props.mass = float(args_cli.presser_mass)
    env_cfg.viewer.eye = (
        tuple(float(v) for v in args_cli.viewer_eye) if args_cli.viewer_eye is not None else (-0.35, -0.35, 0.95)
    )
    env_cfg.viewer.lookat = (
        tuple(float(v) for v in args_cli.viewer_lookat) if args_cli.viewer_lookat is not None else (0.95, 0.0, 0.66)
    )
    configure_tacmap_debug(env_cfg)
    configure_hydroshear_debug(env_cfg)
    configure_taxim_rgb_debug(env_cfg)
    configure_pressure_debug(env_cfg)
    configure_pressure_pad_collision_properties(env_cfg)
    ours_params = ours_rl_component_params(env_cfg)
    rl_local_tacmap_enabled = bool(ours_params.get("local_tacmap_enabled", False))
    if visual_local_tacmap_enabled() and not rl_local_tacmap_enabled:
        focus_index = FINGER_CHOICES.index(str(args_cli.focus_finger))
        focus_link = str(TIANJI_TACMAP_LINK_ORDER[focus_index])
        reference_rows = int(args_cli.tacmap_display_rows)
        reference_cols = int(args_cli.tacmap_display_cols)
        reference_rows = reference_rows if reference_rows > 0 else 240
        reference_cols = reference_cols if reference_cols > 0 else 320
        surface_cfg = _make_tacmap_link_surface_cfg(
            focus_link,
            sensor_kind="surface",
            image_rows=reference_rows,
            image_cols=reference_cols,
        )
        surface_cfg.max_distance = float(TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M)
        surface_cfg.update_period = 1.0e9
        surface_cfg.debug_viz_link_surfaces = False
        surface_cfg.debug_viz_hits = False
        surface_cfg.debug_viz_rays = False
        object_cfg = _make_tacmap_link_surface_cfg(
            focus_link,
            sensor_kind="object",
            image_rows=25,
            image_cols=40,
        )
        object_cfg.max_distance = float(TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M)
        object_cfg.debug_viz_hits = False
        object_cfg.debug_viz_rays = False
        setattr(env_cfg.scene, VISUAL_LOCAL_TACMAP_SURFACE_SENSOR, surface_cfg)
        setattr(env_cfg.scene, VISUAL_LOCAL_TACMAP_OBJECT_SENSOR, object_cfg)
    return env_cfg


def write_phase_pose(
    env,
    phase: str,
    local_step: int,
    alpha: float,
    *,
    tac_link: str,
    tac_center_l: np.ndarray,
    tac_normal_l: np.ndarray,
    tac_slide_axis_l: np.ndarray,
    pressure_link: str,
    pressure_center_l: np.ndarray,
    pressure_normal_l: np.ndarray,
    tac_obj_quat_l: np.ndarray,
) -> None:
    hand_actor = str(args_cli.press_motion_actor).lower() == "hand" and not presser_force_enabled()
    dynamic_axis = dynamic_axis_enabled()
    if phase.startswith("tacmap"):
        press_alpha = 1.0 if phase.endswith("hold") else alpha
        start_offset = float(args_cli.press_start_offset)
        offset = start_offset + press_alpha * (float(args_cli.press_end_offset) - start_offset)
        slide = np.zeros(3, dtype=np.float32)
        slide_target_m = 0.0
        slide_axis_l = tac_slide_axis_l
        if int(args_cli.press_slide_steps) > 0 and float(args_cli.press_slide_distance) != 0.0:
            slide_counter = max(0, int(local_step) - (max(1, int(args_cli.press_steps)) - 1))
            slide_alpha = min(float(slide_counter) / float(max(1, int(args_cli.press_slide_steps))), 1.0)
            slide = tac_slide_axis_l * (float(args_cli.press_slide_distance) * slide_alpha)
            slide_target_m = abs(float(args_cli.press_slide_distance)) * slide_alpha
            slide_axis_l = tac_slide_axis_l * (1.0 if float(args_cli.press_slide_distance) > 0.0 else -1.0)
        if dynamic_axis:
            travel = max(0.0, start_offset - float(args_cli.press_end_offset))
            travel_limit = max(travel, max(0.0, float(args_cli.presser_axis_max_travel)))
            ensure_dynamic_axis_presser(
                env,
                tac_link,
                tac_center_l,
                tac_normal_l,
                tac_obj_quat_l,
                offset_m=start_offset,
                key_base=("tacmap", tac_link),
                force_reset=int(local_step) == 0 and not phase.endswith("hold"),
                drive_sign=1.0 if hand_actor else -1.0,
                max_travel_m=None if hand_actor else travel_limit,
                slide_axis_l_np=slide_axis_l,
                max_slide_m=abs(float(args_cli.press_slide_distance)),
            )
            if not hand_actor:
                drive_dynamic_axis_presser(
                    env,
                    None if presser_force_enabled() else press_alpha * travel,
                    target_slide_m=slide_target_m,
                )
        elif hand_actor:
            ensure_fixed_presser_pose(
                env,
                tac_link,
                tac_center_l,
                tac_normal_l,
                tac_obj_quat_l,
                offset_m=start_offset,
                key_base=("tacmap", tac_link),
                force_reset=int(local_step) == 0 and not phase.endswith("hold"),
            )
        else:
            write_object_pose(
                env,
                tac_link,
                tac_center_l,
                tac_normal_l,
                tac_obj_quat_l,
                offset_m=offset,
                slide_l_np=slide,
            )
    else:
        press_alpha = 1.0 if phase.endswith("hold") else alpha
        start_offset = float(args_cli.pressure_pad_press_start_offset)
        offset = start_offset + press_alpha * (float(args_cli.pressure_pad_press_end_offset) - start_offset)
        if dynamic_axis:
            travel = max(0.0, start_offset - float(args_cli.pressure_pad_press_end_offset))
            travel_limit = max(travel, max(0.0, float(args_cli.presser_axis_max_travel)))
            ensure_dynamic_axis_presser(
                env,
                pressure_link,
                pressure_center_l,
                pressure_normal_l,
                tac_obj_quat_l,
                offset_m=start_offset,
                key_base=("pressure_pad", pressure_link),
                force_reset=int(local_step) == 0 and not phase.endswith("hold"),
                drive_sign=1.0 if hand_actor else -1.0,
                max_travel_m=None if hand_actor else travel_limit,
            )
            if not hand_actor and not presser_force_enabled():
                drive_dynamic_axis_presser(env, press_alpha * travel)
        elif hand_actor:
            ensure_fixed_presser_pose(
                env,
                pressure_link,
                pressure_center_l,
                pressure_normal_l,
                tac_obj_quat_l,
                offset_m=start_offset,
                key_base=("pressure_pad", pressure_link),
                force_reset=int(local_step) == 0 and not phase.endswith("hold"),
            )
        else:
            write_object_pose(
                env,
                pressure_link,
                pressure_center_l,
                pressure_normal_l,
                tac_obj_quat_l,
                offset_m=offset,
            )


def write_scripted_state(
    env,
    robot_hold: RobotHold,
    robot_hold_mode: str,
    phase: str,
    local_step: int,
    alpha: float,
    *,
    tac_link: str,
    tac_center_l: np.ndarray,
    tac_normal_l: np.ndarray,
    tac_slide_axis_l: np.ndarray,
    pressure_link: str,
    pressure_center_l: np.ndarray,
    pressure_normal_l: np.ndarray,
    tac_obj_quat_l: np.ndarray,
) -> None:
    hand_actor = str(args_cli.press_motion_actor).lower() == "hand" and not presser_force_enabled()
    joint_alpha = 1.0 if phase.endswith("hold") else float(alpha)
    joint_target = robot_hold.press_joint_target(joint_alpha) if hand_actor else None
    robot_hold.write(robot_hold_mode, joint_target=joint_target)
    write_phase_pose(
        env,
        phase,
        local_step,
        alpha,
        tac_link=tac_link,
        tac_center_l=tac_center_l,
        tac_normal_l=tac_normal_l,
        tac_slide_axis_l=tac_slide_axis_l,
        pressure_link=pressure_link,
        pressure_center_l=pressure_center_l,
        pressure_normal_l=pressure_normal_l,
        tac_obj_quat_l=tac_obj_quat_l,
    )
    apply_dynamic_axis_presser_force(env)


def step_scripted_visualizer_physics(
    env,
    robot_hold: RobotHold,
    robot_hold_mode: str,
    phase: str,
    local_step: int,
    alpha: float,
    *,
    tac_link: str,
    tac_center_l: np.ndarray,
    tac_normal_l: np.ndarray,
    tac_slide_axis_l: np.ndarray,
    pressure_link: str,
    pressure_center_l: np.ndarray,
    pressure_normal_l: np.ndarray,
    tac_obj_quat_l: np.ndarray,
    update_contact_stats: bool = True,
) -> None:
    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    render_interval = max(1, int(getattr(env.cfg.sim, "render_interval", 1)))
    decimation = max(1, int(getattr(env.cfg, "decimation", 1)))
    for _ in range(decimation):
        env._sim_step_counter += 1
        write_scripted_state(
            env,
            robot_hold,
            robot_hold_mode,
            phase,
            local_step,
            alpha,
            tac_link=tac_link,
            tac_center_l=tac_center_l,
            tac_normal_l=tac_normal_l,
            tac_slide_axis_l=tac_slide_axis_l,
            pressure_link=pressure_link,
            pressure_center_l=pressure_center_l,
            pressure_normal_l=pressure_normal_l,
            tac_obj_quat_l=tac_obj_quat_l,
        )
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        write_scripted_state(
            env,
            robot_hold,
            robot_hold_mode,
            phase,
            local_step,
            alpha,
            tac_link=tac_link,
            tac_center_l=tac_center_l,
            tac_normal_l=tac_normal_l,
            tac_slide_axis_l=tac_slide_axis_l,
            pressure_link=pressure_link,
            pressure_center_l=pressure_center_l,
            pressure_normal_l=pressure_normal_l,
            tac_obj_quat_l=tac_obj_quat_l,
        )
        project_dynamic_axis_presser(env)
        if not dynamic_axis_enabled():
            env.scene["object"].update(0.0)
        env.scene.write_data_to_sim()
        if env._sim_step_counter % render_interval == 0 and is_rendering:
            env.sim.render()
    env.episode_length_buf += 1
    env.common_step_counter += 1
    if hasattr(env, "observation_manager"):
        env.obs_buf = env.observation_manager.compute(update_history=True)
    if bool(update_contact_stats):
        update_focus_contact_stats(env, tac_link if phase.startswith("tacmap") else pressure_link)


def benchmark_mode_enabled() -> bool:
    return int(args_cli.benchmark_steps) > 0


def synchronize_benchmark_cuda(env) -> None:
    if not bool(args_cli.benchmark_sync_cuda) or not torch.cuda.is_available():
        return
    if not str(getattr(env, "device", "")).startswith("cuda"):
        return
    torch.cuda.synchronize(device=env.device)


def _shape_list(value) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(dim) for dim in shape]


def benchmark_observation_shapes(env, env_cfg) -> dict[str, list[int] | None]:
    hydroshear_output = getattr(env, "_rl_hydroshear_output", None)
    obs_buf = getattr(env, "obs_buf", None)
    component_cfg = ours_rl_component_params(env_cfg)
    marker_rows = int(component_cfg.get("hydroshear_marker_rows", 10))
    marker_cols = int(component_cfg.get("hydroshear_marker_cols", 10))
    marker_count = marker_rows * marker_cols
    shapes: dict[str, list[int] | None] = {
        "pressure": _shape_list(getattr(env, "_rl_pressure_observation", None)),
        "depth_m": _shape_list(getattr(env, "_rl_ours_dense_tacmap_depth_m", None)),
        "rgb_flat": _shape_list(getattr(env, "_rl_ours_taxim_rgb_observation", None)),
        "marker_motion": [int(env.num_envs), len(FINGER_CHOICES), marker_count, 3],
        "marker_motion_flat": [int(env.num_envs), len(FINGER_CHOICES) * marker_count * 3],
        "debug_marker_flow": _shape_list(getattr(hydroshear_output, "marker_flow", None)),
    }
    if isinstance(obs_buf, dict):
        for group_name, value in obs_buf.items():
            shapes[f"observation_group/{group_name}"] = _shape_list(value)
    return shapes


def _git_metadata() -> dict[str, str | bool | None]:
    def run(*git_args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *git_args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = completed.stdout.strip()
        return value if completed.returncode == 0 and value else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "short_commit": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(status),
    }


def write_benchmark_result(env, env_cfg, step_times_s: list[float]) -> tuple[Path, dict]:
    times = np.asarray(step_times_s, dtype=np.float64)
    if int(times.size) != int(args_cli.benchmark_steps):
        raise RuntimeError(
            f"Benchmark recorded {times.size} steps, expected {int(args_cli.benchmark_steps)}"
        )
    total_s = float(np.sum(times))
    mean_s = float(np.mean(times))
    sim_hz = 1.0 / mean_s if mean_s > 0.0 else 0.0
    target_sensor_hz = 1.0 / float(env.step_dt)
    num_envs = int(env.num_envs)
    device = str(env.device)
    cuda_active = torch.cuda.is_available() and device.startswith("cuda")
    gpu: dict[str, object] = {
        "device": device,
        "name": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "current_allocated_bytes": None,
        "current_reserved_bytes": None,
    }
    if cuda_active:
        properties = torch.cuda.get_device_properties(device)
        gpu.update(
            {
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "revo3_full_tactile_observation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_metadata(),
        "configuration": {
            "task": str(args_cli.task),
            "num_envs": num_envs,
            "warmup_steps": int(args_cli.benchmark_warmup_steps),
            "measured_steps": int(times.size),
            "sync_cuda_per_step": bool(args_cli.benchmark_sync_cuda),
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(env_cfg.decimation),
            "control_dt_s": float(env.step_dt),
            "target_sensor_hz": target_sensor_hz,
            "focus_finger": str(args_cli.focus_finger),
            "presser": str(args_cli.presser),
            "target": str(args_cli.target),
            "presser_force_n": float(args_cli.presser_force_n),
            "headless": bool(args_cli.headless),
            "visualization_in_timed_region": False,
            "modalities": ["pressure", "depth", "rgb", "marker_motion"],
        },
        "metrics": {
            "total_time_s": total_s,
            "mean_step_s": mean_s,
            "median_step_s": float(np.median(times)),
            "std_step_s": float(np.std(times)),
            "min_step_s": float(np.min(times)),
            "p95_step_s": float(np.percentile(times, 95)),
            "p99_step_s": float(np.percentile(times, 99)),
            "max_step_s": float(np.max(times)),
            "sim_hz": sim_hz,
            "env_steps_per_s": float(num_envs) * sim_hz,
            "physics_steps_per_s": float(env_cfg.decimation) * sim_hz,
            "real_time_factor": sim_hz / target_sensor_hz,
        },
        "gpu": gpu,
        "observation_shapes": benchmark_observation_shapes(env, env_cfg),
    }
    output_path = args_cli.benchmark_output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            REPO_ROOT
            / "outputs"
            / "revo3_tactile_benchmark"
            / f"{timestamp}__envs-{num_envs:04d}.json"
        )
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, result


def main() -> None:
    if args_cli.num_envs != 1 and not benchmark_mode_enabled():
        print("[WARN] This visualizer displays env_0 only; running more envs is allowed but wasteful.", flush=True)
    if int(args_cli.benchmark_steps) < 0:
        raise ValueError("--benchmark-steps must be non-negative.")
    if int(args_cli.benchmark_warmup_steps) < 0:
        raise ValueError("--benchmark-warmup-steps must be non-negative.")
    if args_cli.benchmark_output is not None and not benchmark_mode_enabled():
        raise ValueError("--benchmark-output requires a positive --benchmark-steps value.")
    if bool(args_cli.pressure_display_diffusion):
        if not np.isfinite(float(args_cli.pressure_display_diffusion_sigma_mm)) or float(
            args_cli.pressure_display_diffusion_sigma_mm
        ) <= 0.0:
            raise ValueError("--pressure-display-diffusion-sigma-mm must be finite and positive.")
        if not np.isfinite(float(args_cli.pressure_display_diffusion_blend)) or not 0.0 <= float(
            args_cli.pressure_display_diffusion_blend
        ) <= 1.0:
            raise ValueError("--pressure-display-diffusion-blend must be within [0, 1].")
        if not np.isfinite(float(args_cli.pressure_display_diffusion_radius_sigma)) or float(
            args_cli.pressure_display_diffusion_radius_sigma
        ) <= 0.0:
            raise ValueError("--pressure-display-diffusion-radius-sigma must be finite and positive.")
        if not np.isfinite(float(args_cli.pressure_display_diffusion_normal_power)) or float(
            args_cli.pressure_display_diffusion_normal_power
        ) < 0.0:
            raise ValueError("--pressure-display-diffusion-normal-power must be finite and non-negative.")
    if dynamic_axis_enabled():
        if not bool(args_cli.presser_collision_enabled):
            raise ValueError("--presser-body-mode dynamic_axis requires --enable-presser-collision.")
    if float(args_cli.presser_force_n) > 0.0:
        if not dynamic_axis_enabled():
            raise ValueError("--presser-force-n requires --presser-body-mode dynamic_axis.")
        if not bool(args_cli.presser_collision_enabled):
            raise ValueError("--presser-force-n requires --enable-presser-collision.")
    if int(args_cli.presser_force_ramp_steps) < 0:
        raise ValueError("--presser-force-ramp-steps must be non-negative.")
    if not np.isfinite(float(args_cli.hydroshear_marker_motion_scale)) or float(
        args_cli.hydroshear_marker_motion_scale
    ) <= 0.0:
        raise ValueError("--hydroshear-marker-motion-scale must be finite and positive.")
    if not np.isfinite(float(args_cli.camera_frame_axis_length)) or float(args_cli.camera_frame_axis_length) <= 0.0:
        raise ValueError("--camera-frame-axis-length must be finite and positive.")
    if not np.isfinite(float(args_cli.camera_frame_axis_width)) or float(args_cli.camera_frame_axis_width) <= 0.0:
        raise ValueError("--camera-frame-axis-width must be finite and positive.")
    if int(args_cli.camera_visible_surface_samples) <= 0:
        raise ValueError("--camera-visible-surface-samples must be positive.")
    if int(args_cli.camera_visible_boundary_points) < 3:
        raise ValueError("--camera-visible-boundary-points must be at least 3.")
    if not np.isfinite(float(args_cli.tacmap_local_contact_threshold_mm)) or float(
        args_cli.tacmap_local_contact_threshold_mm
    ) < 0.0:
        raise ValueError("--tacmap-local-contact-threshold-mm must be finite and non-negative.")
    if not np.isfinite(float(args_cli.tacmap_local_roi_margin_mm)) or float(
        args_cli.tacmap_local_roi_margin_mm
    ) < 0.0:
        raise ValueError("--tacmap-local-roi-margin-mm must be finite and non-negative.")
    if not np.isfinite(float(args_cli.tacmap_local_roi_guard_mm)) or float(
        args_cli.tacmap_local_roi_guard_mm
    ) < 0.0:
        raise ValueError("--tacmap-local-roi-guard-mm must be finite and non-negative.")
    if int(args_cli.tacmap_local_roi_stable_frames) <= 0:
        raise ValueError("--tacmap-local-roi-stable-frames must be positive.")
    if not np.isfinite(float(args_cli.local_tacmap_point_radius)) or float(
        args_cli.local_tacmap_point_radius
    ) <= 0.0:
        raise ValueError("--local-tacmap-point-radius must be finite and positive.")
    if int(args_cli.visual_update_interval) <= 0:
        raise ValueError("--visual-update-interval must be positive.")
    if not np.isfinite(float(args_cli.taxim_rgb_depth_scale)) or float(args_cli.taxim_rgb_depth_scale) <= 0.0:
        raise ValueError("--taxim-rgb-depth-scale must be finite and positive.")
    for name, value in (
        ("--press-steps", args_cli.press_steps),
        ("--pressure-pad-press-steps", args_cli.pressure_pad_press_steps),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")

    if bool(args_cli.record_side_view):
        if int(args_cli.side_view_resolution) <= 0:
            raise ValueError("--side-view-resolution must be positive.")
        if not np.isfinite(float(args_cli.side_view_distance)) or float(args_cli.side_view_distance) <= 0.0:
            raise ValueError("--side-view-distance must be finite and positive.")
        if not np.isfinite(float(args_cli.side_view_elevation)):
            raise ValueError("--side-view-elevation must be finite.")

    env_cfg = configure_env()
    # Keep the object point-cloud observation, but do not render its red surface markers.
    env_cfg.observations.perception.object_point_cloud.params["visualize"] = False
    if bool(args_cli.record_side_view):
        env_cfg.viewer.resolution = (int(args_cli.side_view_resolution), int(args_cli.side_view_resolution))
        env_cfg.commands.object_pose.debug_vis = False
    gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if bool(args_cli.record_side_view) else None,
    )
    env = gym_env.unwrapped
    object_pose_term = env.command_manager.get_term("object_pose")
    current_body_pose_visualizer = getattr(object_pose_term, "curr_visualizer", None)
    if current_body_pose_visualizer is not None:
        current_body_pose_visualizer.set_visibility(False)
    env.reset()
    initialize_visual_local_tacmap_refinement(env)
    taxim_rgb_component_cfg = ours_rl_component_params(env_cfg) if bool(args_cli.show_taxim_rgb) else None
    press_joint_phase = "pressure_pad" if args_cli.target == "pressure_pad" else "tacmap"
    robot_hold = RobotHold(
        env,
        focus_finger=str(args_cli.focus_finger),
        press_joint_names=default_press_joint_names(str(args_cli.focus_finger), press_joint_phase),
        sharpa_finger_target_gain=float(args_cli.sharpa_finger_target_gain),
    )
    tac_link, tac_center_l, tac_normal_l, tac_slide_axis_l, tac_tilt_quat_l = tacmap_target(args_cli.focus_finger)
    tac_obj_quat_l = np.asarray(
        quat_mul_tuple(
            tuple(float(v) for v in tac_tilt_quat_l),
            tuple(float(v) for v in presser_quat_l()),
        ),
        dtype=np.float32,
    )
    if bool(args_cli.record_side_view) and args_cli.viewer_eye is None and args_cli.viewer_lookat is None:
        robot: Articulation = env.scene["robot"]
        link_idx = body_index(robot, tac_link)
        link_state = robot.data.body_link_state_w[0, link_idx, :7]
        center_l = torch.as_tensor(tac_center_l, device=env.device, dtype=torch.float32)
        normal_l = torch.as_tensor(tac_normal_l, device=env.device, dtype=torch.float32)
        center_w = link_state[:3] + quat_apply(link_state[3:7].unsqueeze(0), center_l.unsqueeze(0))[0]
        normal_w = quat_apply(link_state[3:7].unsqueeze(0), normal_l.unsqueeze(0))[0]
        normal_w = normal_w / torch.clamp(torch.linalg.norm(normal_w), min=1.0e-8)
        world_up = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=torch.float32)
        side_w = torch.linalg.cross(normal_w, world_up, dim=0)
        if float(torch.linalg.norm(side_w).detach().cpu()) < 1.0e-6:
            side_w = torch.linalg.cross(
                normal_w,
                torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=torch.float32),
                dim=0,
            )
        side_w = -side_w / torch.clamp(torch.linalg.norm(side_w), min=1.0e-8)
        eye_w = (
            center_w
            + side_w * float(args_cli.side_view_distance)
            + world_up * float(args_cli.side_view_elevation)
        )
        eye = eye_w.detach().cpu().tolist()
        target = center_w.detach().cpu().tolist()
        env.sim.set_camera_view(eye=eye, target=target)
        print(
            "[INFO] Automatic TacMap side view: "
            f"--viewer-eye {' '.join(f'{value:.6f}' for value in eye)} "
            f"--viewer-lookat {' '.join(f'{value:.6f}' for value in target)}",
            flush=True,
        )
    if bool(args_cli.record_side_view):
        for _ in range(3):
            env.render()
    pressure_link, pressure_center_l, pressure_normal_l = pressure_pad_target(
        pressure_pad_link_name(args_cli.focus_finger, args_cli.pressure_pad_segment)
    )
    if str(args_cli.tacmap_press_tilt_axis).lower() != "none" and abs(float(args_cli.tacmap_press_tilt_deg)) > 1.0e-12:
        print(
            "[INFO] TacMap press tilt: "
            f"finger={args_cli.focus_finger} axis={args_cli.tacmap_press_tilt_axis} "
            f"deg={float(args_cli.tacmap_press_tilt_deg):g} "
            f"center_l={tuple(float(v) for v in tac_center_l)} normal_l={tuple(float(v) for v in tac_normal_l)}",
            flush=True,
        )
    env._rl_tactile_viz_tacmap_link = tac_link
    env._rl_tactile_viz_pressure_link = pressure_link
    panel = None
    contact_panel = None
    calibrated_marker_viz = None
    selected_surface_point_viz = None
    hydroshear_roi_viz = None
    hydroshear_tacmap_point_viz = None
    hydroshear_roi_point_viz = None
    local_tacmap_point_viz = None
    local_tacmap_object_ray_viz = None
    local_tacmap_reference_ray_viz = None
    local_tacmap_reference_boundary_count = 0
    camera_frame_viz: list[LineSegmentViz] = []
    camera_frustum_surface_boundary_viz = None
    camera_frustum_camera_xy_boundary_viz = None
    camera_frustum_camera_xy_rectangle_viz = None
    camera_visible_surface_boundary_viz = None
    camera_visible_plane_boundary_viz = None
    camera_visible_plane_rectangle_viz = None
    pressure_taxel_cache = pressure_pad_taxel_local_cache()
    calibrated_marker_cache = (
        calibrated_marker_local_cache(focus_finger=str(args_cli.focus_finger))
        if bool(args_cli.show_calibrated_marker_points)
        else []
    )
    selected_surface_marker_mask = None
    if bool(args_cli.show_surface_debug and args_cli.show_selected_surface_points):
        with np.load(VITAI_MARKER_LAYOUT, allow_pickle=False) as layout:
            selected_surface_marker_mask = np.asarray(
                layout[f"{args_cli.focus_finger}_ray_is_marker"],
                dtype=bool,
            ).reshape(TIANJI_TACMAP_RL_ROWS, TIANJI_TACMAP_RL_COLS)
    camera_frame_local_cache = (
        calibrated_camera_frame_local_cache(
            str(args_cli.focus_finger),
            axis_length_m=float(args_cli.camera_frame_axis_length),
        )
        if bool(args_cli.show_surface_debug and args_cli.show_camera_frame)
        else []
    )
    camera_frustum_surface_boundary_cache = (
        camera_frustum_surface_boundary_local_cache(
            focus_finger=str(args_cli.focus_finger),
            surface_offset_m=float(args_cli.camera_visible_point_radius) * 2.0,
        )
        if bool(args_cli.show_surface_debug and args_cli.show_camera_visible_surface)
        else []
    )
    camera_frustum_exact_boundary_cache = (
        camera_frustum_surface_boundary_local_cache(
            focus_finger=str(args_cli.focus_finger),
            surface_offset_m=0.0,
            print_summary=False,
        )
        if camera_frustum_surface_boundary_cache
        else []
    )
    camera_frustum_camera_xy_boundary_cache = (
        camera_frustum_boundary_camera_xy_local_cache(
            camera_frustum_exact_boundary_cache,
            focus_finger=str(args_cli.focus_finger),
        )
        if camera_frustum_exact_boundary_cache
        else []
    )
    camera_frustum_camera_xy_rectangle_cache = (
        calibrated_camera_ray_rectangle_local_cache(
            focus_finger=str(args_cli.focus_finger),
        )
        if bool(args_cli.show_surface_debug and args_cli.show_camera_visible_surface)
        else []
    )
    camera_visible_surface_caches = (
        camera_visible_surface_local_cache(
            int(args_cli.camera_visible_surface_samples),
            float(args_cli.camera_visible_point_radius),
            focus_finger=str(args_cli.focus_finger),
            boundary_point_count=int(args_cli.camera_visible_boundary_points),
        )[1:]
        if bool(args_cli.show_surface_debug and args_cli.show_camera_visible_surface)
        else ([], [], [])
    )
    (
        camera_visible_surface_boundary_cache,
        camera_visible_plane_boundary_cache,
        _legacy_camera_visible_plane_rectangle_cache,
    ) = camera_visible_surface_caches
    # The old red hull-fitted rectangle is intentionally hidden. The shared
    # white rectangle above is now the only dense-ray source.
    camera_visible_plane_rectangle_cache: list[tuple[str, np.ndarray]] = []
    pressure_display_diffusion_kernels = None
    if bool(args_cli.pressure_display_diffusion):
        sigma_m = float(args_cli.pressure_display_diffusion_sigma_mm) * 1.0e-3
        pressure_display_diffusion_kernels = {
            str(link_name): build_pressure_display_diffusion_kernel(
                points_l,
                normals_l,
                sigma_m=sigma_m,
                radius_sigma=float(args_cli.pressure_display_diffusion_radius_sigma),
                normal_power=float(args_cli.pressure_display_diffusion_normal_power),
            )
            for link_name, points_l, normals_l in pressure_taxel_cache
        }
        print(
            "[INFO] Pressure display-only Gaussian diffusion enabled on CPU: "
            f"sigma={float(args_cli.pressure_display_diffusion_sigma_mm):g}mm "
            f"blend={float(args_cli.pressure_display_diffusion_blend):g} "
            f"radius={float(args_cli.pressure_display_diffusion_radius_sigma):g}sigma "
            f"normal_power={float(args_cli.pressure_display_diffusion_normal_power):g}; "
            "RL pressure observation remains raw.",
            flush=True,
        )
    if bool(args_cli.show_surface_debug and args_cli.show_hydroshear_roi):
        hydroshear_roi_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/HydroShearROI/BoundaryWireframeOrange",
            color=(1.0, 0.35, 0.0),
            line_width=max(1.0e-7, float(args_cli.hydroshear_roi_line_width)),
        )
        hydroshear_tacmap_point_viz = PointNormalViz(
            "/Visuals/RLTactileObs/HydroShearROI/GlobalTacMap32x24SurfaceHits",
            point_color=(0.0, 0.85, 1.0),
            normal_color=(1.0, 1.0, 1.0),
            point_radius=max(1.0e-7, float(args_cli.hydroshear_tacmap_point_radius)),
            line_width=float(args_cli.surface_debug_ray_width),
            normal_length=0.0,
        )
        hydroshear_roi_point_viz = PointNormalViz(
            "/Visuals/RLTactileObs/HydroShearROI/SelectedPoissonSamples",
            point_color=(1.0, 0.82, 0.0),
            normal_color=(1.0, 1.0, 1.0),
            point_radius=max(1.0e-7, float(args_cli.hydroshear_roi_point_radius)),
            line_width=float(args_cli.surface_debug_ray_width),
            normal_length=0.0,
        )
    if bool(args_cli.show_surface_debug) and calibrated_marker_cache:
        # Keep calibrated markers separate from the optional dense selected-hit layer.
        calibrated_marker_viz = PointNormalViz(
            "/Visuals/RLTactileObs/CalibratedVitaiMarkers",
            point_color=(0.0, 0.2, 1.0),
            normal_color=(1.0, 1.0, 0.0),
            point_radius=float(args_cli.hydroshear_marker_point_radius),
            line_width=float(args_cli.hydroshear_marker_normal_width),
            normal_length=float(args_cli.hydroshear_marker_normal_length),
        )
    if selected_surface_marker_mask is not None:
        selected_surface_point_viz = PointNormalViz(
            "/Visuals/RLTactileObs/TacMapSelectedRubberSurfacePoints",
            point_color=(0.0, 0.85, 1.0),
            normal_color=(1.0, 1.0, 1.0),
            point_radius=float(args_cli.selected_surface_point_radius),
            line_width=float(args_cli.surface_debug_ray_width),
            normal_length=0.0,
        )
    if (
        bool(args_cli.show_surface_debug and args_cli.show_local_tacmap_points)
        and isinstance(getattr(env, "_rl_visual_local_tacmap_refinement", None), dict)
    ):
        local_tacmap_point_viz = PointNormalViz(
            "/Visuals/RLTactileObs/LocalTacMapSurfaceSamples",
            point_color=(1.0, 0.55, 0.0),
            normal_color=(1.0, 1.0, 1.0),
            point_radius=float(args_cli.local_tacmap_point_radius),
            line_width=float(args_cli.surface_debug_ray_width),
            normal_length=0.0,
        )
    if (
        bool(args_cli.show_surface_debug and args_cli.debug_sensors)
        and isinstance(getattr(env, "_rl_visual_local_tacmap_refinement", None), dict)
    ):
        local_tacmap_object_ray_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/LocalTacMapRays/Selected1000ObjectRaysYellow",
            color=(1.0, 0.85, 0.0),
            line_width=float(args_cli.surface_debug_ray_width),
        )
        local_tacmap_reference_ray_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/LocalTacMapRays/DenseReferenceBoundaryCyan",
            color=(0.0, 0.85, 1.0),
            line_width=float(args_cli.surface_debug_ray_width),
        )
        local_tacmap_state = env._rl_visual_local_tacmap_refinement
        local_tacmap_reference_boundary_count = int(
            local_tacmap_state["reference_boundary_indices"].numel()
        )
    if camera_visible_surface_boundary_cache:
        # The surface samples are used once to estimate this hull and are not rendered.
        camera_visible_surface_boundary_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CameraVisibleRubberSurface/BoundaryGreen",
            color=(0.0, 1.0, 0.15),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius)),
        )
    if camera_frustum_surface_boundary_cache:
        camera_frustum_surface_boundary_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CameraVisibleRubberSurface/CameraRangeAug4BoundaryRed",
            color=(1.0, 0.0, 0.0),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius) * 2.0),
        )
    if camera_frustum_camera_xy_boundary_cache:
        camera_frustum_camera_xy_boundary_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CalibratedCameraFrame/CameraRangeAug4BoundaryXYPlaneMagenta",
            color=(1.0, 0.0, 1.0),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius) * 2.0),
        )
    if camera_frustum_camera_xy_rectangle_cache:
        camera_frustum_camera_xy_rectangle_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CalibratedCameraFrame/CameraRangeAug4MinimumRectangle320x240White",
            color=(1.0, 1.0, 1.0),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius) * 2.0),
        )
    if camera_visible_plane_boundary_cache:
        # Yellow is the same visible boundary orthogonally projected along
        # camera Z onto the calibrated camera XY plane at Z=0.
        camera_visible_plane_boundary_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CalibratedCameraFrame/VisibleBoundaryXYPlaneYellow",
            color=(1.0, 1.0, 0.0),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius)),
        )
    if camera_visible_plane_rectangle_cache:
        # Red is the smallest fixed-aspect (320:240 or 240:320) rectangle that
        # fully encloses the yellow calibrated camera-XY visible boundary.
        camera_visible_plane_rectangle_viz = LineSegmentViz(
            "/Visuals/RLTactileObs/CalibratedCameraFrame/MinimumEnclosingRectangleXYPlaneRed",
            color=(1.0, 0.0, 0.0),
            line_width=max(1.0e-7, float(args_cli.camera_visible_point_radius)),
        )
    if camera_frame_local_cache:
        for axis_name, axis_color in (
            ("XRed", (1.0, 0.0, 0.0)),
            ("YGreen", (0.0, 1.0, 0.0)),
            ("ZBlue", (0.0, 0.0, 1.0)),
        ):
            camera_frame_viz.append(
                LineSegmentViz(
                    f"/Visuals/RLTactileObs/CalibratedCameraFrame/{axis_name}",
                    color=axis_color,
                    line_width=float(args_cli.camera_frame_axis_width),
                )
            )
    debug_line_cache = (
        camera_frame_local_cache
        + camera_frustum_surface_boundary_cache
        + camera_frustum_camera_xy_boundary_cache
        + camera_frustum_camera_xy_rectangle_cache
        + camera_visible_surface_boundary_cache
        + camera_visible_plane_boundary_cache
        + camera_visible_plane_rectangle_cache
    )
    camera_axis_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_frame_local_cache
    )
    frustum_surface_boundary_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_frustum_surface_boundary_cache
    )
    frustum_camera_xy_boundary_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_frustum_camera_xy_boundary_cache
    )
    frustum_camera_xy_rectangle_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_frustum_camera_xy_rectangle_cache
    )
    surface_boundary_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_visible_surface_boundary_cache
    )
    plane_boundary_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_visible_plane_boundary_cache
    )
    plane_rectangle_count = sum(
        np.asarray(segments_l).reshape(-1, 2, 3).shape[0]
        for _link_name, segments_l in camera_visible_plane_rectangle_cache
    )
    step = 0
    image: np.ndarray | None = None
    stats = "waiting for first RL observation"
    benchmark_enabled = benchmark_mode_enabled()
    benchmark_step_times_s: list[float] = []
    if benchmark_enabled:
        print(
            "[BENCHMARK] Revo3 full tactile observation: "
            f"envs={int(env.num_envs)} warmup={int(args_cli.benchmark_warmup_steps)} "
            f"measure={int(args_cli.benchmark_steps)} sync_cuda={bool(args_cli.benchmark_sync_cuda)} "
            "modalities=pressure,depth,rgb,marker_motion visualization=excluded",
            flush=True,
        )
    video_encoder = None
    record_path = None if args_cli.record_video is None else Path(args_cli.record_video).expanduser().resolve()
    if record_path is not None and record_path.suffix.lower() != ".mp4":
        raise ValueError("--record-video output must use the .mp4 extension.")
    if float(args_cli.record_fps) < 0.0:
        raise ValueError("--record-fps must be non-negative.")

    print(
        "[INFO] RL tactile obs visualizer: "
        f"task={args_cli.task}, presser={args_cli.presser}, target={args_cli.target}, "
        f"tacmap_link={tac_link}, pressure_link={pressure_link}, "
        f"press_motion_actor={args_cli.press_motion_actor}, "
        f"presser_body_mode={args_cli.presser_body_mode}, "
        f"presser_collision={'on' if args_cli.presser_collision_enabled else 'off'}, "
        f"robot_hold_mode={args_cli.robot_hold_mode}, "
        f"tacmap_contact_shell="
        f"{float(ours_rl_component_params(env_cfg).get('tacmap_contact_shell_m', 0.0)) * 1000.0:.3f}mm, "
        f"surface_debug={'on' if args_cli.show_surface_debug else 'off'}",
        f"calibrated_markers={'on' if calibrated_marker_cache else 'off'}",
        f"selected_surface_points={'on' if selected_surface_point_viz is not None else 'off'}",
        f"local_tacmap_{VISUAL_LOCAL_TACMAP_RAY_COUNT}_points="
        f"{'on' if local_tacmap_point_viz is not None else 'off'}",
        f"local_tacmap_object_rays={'on' if local_tacmap_object_ray_viz is not None else 'off'}",
        f"dense_reference_boundary_rays={local_tacmap_reference_boundary_count}",
        f"camera_frame={'on' if camera_frame_viz else 'off'}",
        f"camera_range_aug4_surface_boundary="
        f"{'on' if camera_frustum_surface_boundary_cache else 'off'}",
        f"camera_range_aug4_camera_xy_projection="
        f"{'on' if camera_frustum_camera_xy_boundary_cache else 'off'}",
        f"camera_range_aug4_minimum_rectangle_320x240="
        f"{'on' if camera_frustum_camera_xy_rectangle_cache else 'off'}",
        f"camera_visible_surface_boundary={'on' if camera_visible_surface_boundary_cache else 'off'}",
        f"camera_visible_xy_boundary={'on' if camera_visible_plane_boundary_cache else 'off'}",
        f"camera_visible_minimum_enclosing_rectangle="
        f"{'on' if camera_visible_plane_rectangle_cache else 'off'}",
        f"hydroshear_marker={'on' if args_cli.show_hydroshear_marker else 'off'}",
        f"hydroshear_roi={'on' if args_cli.show_hydroshear_roi else 'off'}",
        f"taxim_rgb={'on' if taxim_rgb_component_cfg is not None else 'off'}",
        f"visual_update_interval={int(args_cli.visual_update_interval)}",
        f"tacmap_display=rot90:{int(args_cli.tacmap_display_rot90) % 4}/"
        f"flip_x:{int(bool(args_cli.tacmap_display_flip_x))}/"
        f"flip_y:{int(bool(args_cli.tacmap_display_flip_y))}",
        flush=True,
    )

    had_tacmap_reading = focus_tacmap_is_active(env)
    while simulation_app.is_running():
        start_time = time.time()
        phase, local_step, alpha = scripted_target(step)
        env._rl_tactile_viz_phase = phase
        env._rl_tactile_viz_global_step = step
        benchmark_measure_this_step = benchmark_enabled and step >= int(args_cli.benchmark_warmup_steps)
        if benchmark_enabled and step == int(args_cli.benchmark_warmup_steps):
            synchronize_benchmark_cuda(env)
            if torch.cuda.is_available() and str(env.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device=env.device)
        if benchmark_measure_this_step:
            synchronize_benchmark_cuda(env)
            benchmark_step_start = time.perf_counter()
        step_scripted_visualizer_physics(
            env,
            robot_hold,
            str(args_cli.robot_hold_mode),
            phase,
            local_step,
            alpha,
            tac_link=tac_link,
            tac_center_l=tac_center_l,
            tac_normal_l=tac_normal_l,
            tac_slide_axis_l=tac_slide_axis_l,
            pressure_link=pressure_link,
            pressure_center_l=pressure_center_l,
            pressure_normal_l=pressure_normal_l,
            tac_obj_quat_l=tac_obj_quat_l,
            update_contact_stats=not benchmark_enabled,
        )
        if benchmark_measure_this_step:
            synchronize_benchmark_cuda(env)
            benchmark_step_times_s.append(time.perf_counter() - benchmark_step_start)
        if benchmark_enabled:
            step += 1
            if len(benchmark_step_times_s) >= int(args_cli.benchmark_steps):
                break
            continue
        visual_update_due = step % int(args_cli.visual_update_interval) == 0
        if visual_update_due and bool(args_cli.show_surface_debug):
            if (
                selected_surface_point_viz is not None
                and selected_surface_marker_mask is not None
            ):
                surface_arrays = focus_selected_surface_point_arrays(
                    env,
                    selected_surface_marker_mask,
                )
                if surface_arrays is not None:
                    points_w, valid, point_colors = surface_arrays
                    selected_surface_point_viz.update(
                        points_w,
                        None,
                        valid=valid,
                        point_colors=point_colors,
                        point_max=0,
                    )
            if calibrated_marker_viz is not None:
                arrays = calibrated_marker_world_arrays(
                    env,
                    calibrated_marker_cache,
                )
                if arrays is not None:
                    points_w, _normals_w, point_colors = arrays
                    calibrated_marker_viz.update(
                        points_w,
                        None,
                        point_colors=point_colors,
                        point_max=0,
                    )
            if debug_line_cache:
                debug_segments_w = local_line_segments_world(env, debug_line_cache)
                segment_cursor = 0
                camera_segments_w = debug_segments_w[
                    segment_cursor : segment_cursor + camera_axis_count
                ]
                segment_cursor += camera_axis_count
            else:
                camera_segments_w = np.zeros((0, 2, 3), dtype=np.float32)
                segment_cursor = 0
            if camera_frame_viz:
                if camera_segments_w.shape == (3, 2, 3):
                    for axis_viz, axis_segment_w in zip(
                        camera_frame_viz,
                        camera_segments_w,
                        strict=True,
                    ):
                        axis_viz.update(axis_segment_w.reshape(1, 2, 3))
            if camera_frustum_surface_boundary_viz is not None:
                camera_frustum_surface_boundary_viz.update(
                    debug_segments_w[
                        segment_cursor : segment_cursor + frustum_surface_boundary_count
                    ]
                )
                segment_cursor += frustum_surface_boundary_count
            if camera_frustum_camera_xy_boundary_viz is not None:
                camera_frustum_camera_xy_boundary_viz.update(
                    debug_segments_w[
                        segment_cursor : segment_cursor + frustum_camera_xy_boundary_count
                    ]
                )
                segment_cursor += frustum_camera_xy_boundary_count
            if camera_frustum_camera_xy_rectangle_viz is not None:
                camera_frustum_camera_xy_rectangle_viz.update(
                    debug_segments_w[
                        segment_cursor : segment_cursor + frustum_camera_xy_rectangle_count
                    ]
                )
                segment_cursor += frustum_camera_xy_rectangle_count
            if camera_visible_surface_boundary_viz is not None:
                camera_visible_surface_boundary_viz.update(
                    debug_segments_w[segment_cursor : segment_cursor + surface_boundary_count]
                )
                segment_cursor += surface_boundary_count
            if camera_visible_plane_boundary_viz is not None:
                camera_visible_plane_boundary_viz.update(
                    debug_segments_w[segment_cursor : segment_cursor + plane_boundary_count]
                )
                segment_cursor += plane_boundary_count
            if camera_visible_plane_rectangle_viz is not None:
                camera_visible_plane_rectangle_viz.update(
                    debug_segments_w[segment_cursor : segment_cursor + plane_rectangle_count]
                )
            if hydroshear_roi_viz is not None:
                roi_segments_w = hydroshear_roi_wireframe_arrays(env)
                if roi_segments_w is None:
                    roi_segments_w = np.zeros((0, 2, 3), dtype=np.float32)
                hydroshear_roi_viz.update(roi_segments_w)
                env._rl_hydroshear_roi_line_count = int(roi_segments_w.shape[0])
            if hydroshear_tacmap_point_viz is not None:
                tacmap_arrays = hydroshear_global_tacmap_surface_debug_arrays(env)
                if tacmap_arrays is None:
                    tacmap_points_w = np.zeros((0, 3), dtype=np.float32)
                    tacmap_valid = np.zeros((0,), dtype=bool)
                    tacmap_colors = np.zeros((0, 3), dtype=np.float32)
                else:
                    tacmap_points_w, tacmap_valid, tacmap_colors = tacmap_arrays
                hydroshear_tacmap_point_viz.update(
                    tacmap_points_w,
                    None,
                    valid=tacmap_valid,
                    point_colors=tacmap_colors,
                    point_max=0,
                )
                env._rl_hydroshear_tacmap_point_count = int(np.count_nonzero(tacmap_valid))
            if hydroshear_roi_point_viz is not None:
                roi_arrays = hydroshear_selected_roi_debug_arrays(env)
                if roi_arrays is None:
                    roi_points_w = np.zeros((0, 3), dtype=np.float32)
                    roi_valid = np.zeros((0,), dtype=bool)
                    roi_colors = np.zeros((0, 3), dtype=np.float32)
                else:
                    roi_points_w, roi_valid, roi_colors = roi_arrays
                hydroshear_roi_point_viz.update(
                    roi_points_w,
                    None,
                    valid=roi_valid,
                    point_colors=roi_colors,
                    point_max=0,
                )
                env._rl_hydroshear_roi_selected_count = int(np.count_nonzero(roi_valid))
                env._rl_hydroshear_roi_contact_count = int(
                    np.count_nonzero(roi_valid & (roi_colors[:, 0] > 0.9) & (roi_colors[:, 1] < 0.2))
                )
                adapter = getattr(env, "_brainco_rl_hydroshear_adapter", None)
                adapter_cfg = None if adapter is None else getattr(adapter, "cfg", None)
                env._rl_hydroshear_roi_capacity = int(
                    getattr(adapter_cfg, "object_sample_roi_count", 0)
                )
        update_visual_local_tacmap_refinement(env)
        if local_tacmap_point_viz is not None:
            arrays = visual_local_tacmap_world_arrays(env)
            if arrays is not None:
                points_w, point_colors = arrays
                local_tacmap_point_viz.update(
                    points_w,
                    None,
                    point_colors=point_colors,
                    point_max=0,
                )
        if local_tacmap_object_ray_viz is not None or local_tacmap_reference_ray_viz is not None:
            ray_arrays = visual_local_tacmap_ray_world_arrays(env)
            if ray_arrays is not None:
                object_ray_segments_w, reference_ray_segments_w = ray_arrays
                if local_tacmap_object_ray_viz is not None:
                    local_tacmap_object_ray_viz.update(object_ray_segments_w)
                if local_tacmap_reference_ray_viz is not None:
                    local_tacmap_reference_ray_viz.update(reference_ray_segments_w)
        if visual_update_due:
            taxim_image, taxim_stats = render_visual_taxim_rgb(env, taxim_rgb_component_cfg)
            image, stats = build_image(
                env,
                pressure_taxel_cache,
                pressure_display_diffusion_kernels,
                taxim_image=taxim_image,
            )
            stats = f"{stats} {taxim_stats}"
            if bool(args_cli.record_side_view):
                side_view = env.render()
                if side_view is not None:
                    image = tile_rgb([side_view, image], cols=2, gap=8, preserve_scale=True)
        assert image is not None
        if record_path is not None:
            frame = to_numpy_uint8_rgb(image)
            if video_encoder is None:
                record_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                fps = float(args_cli.record_fps) or 1.0 / float(env.step_dt)
                video_encoder = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-video_size",
                        f"{int(width)}x{int(height)}",
                        "-framerate",
                        f"{fps:.9g}",
                        "-i",
                        "-",
                        "-an",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-tag:v",
                        "avc1",
                        "-movflags",
                        "+faststart",
                        str(record_path),
                    ],
                    stdin=subprocess.PIPE,
                )
                print(f"[INFO] Recording combined tactile observations to {record_path} at {fps:.3f} FPS.", flush=True)
            if video_encoder.stdin is None:
                raise RuntimeError(f"Unable to open FFmpeg input pipe for {record_path}")
            video_encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        contact_image = None
        if visual_update_due and bool(args_cli.show_tacmap_contact_window) and not args_cli.no_local_ui:
            contact_image = build_tacmap_contact_image(env)
        if args_cli.debug_sensors and int(args_cli.print_every) > 0 and step % int(args_cli.print_every) == 0:
            debug_tacmap_sensors(env, step=step)
            debug_pressure_sensors(env, step=step, pressure_link=pressure_link)
        if panel is None and not args_cli.no_local_ui:
            panel = ObsPanel(
                image,
                title=(
                    "RL Observation — Whole-Hand Pressure (285 Taxels)"
                    if bool(args_cli.whole_hand_pressure_only)
                    else (
                        "RL Observation — Focus Pressure Pad"
                        if bool(args_cli.focus_pressure_only)
                        else (
                            "RL Observation — Focus Depth / RGB / Marker"
                            if bool(args_cli.focus_visuotactile_only)
                            else "RL Observation — Pressure / TacMap / HydroShear"
                        )
                    )
                ),
            )
        if panel is not None and visual_update_due:
            panel.update(image, stats)
        if contact_image is not None and contact_panel is None:
            contact_panel = ObsPanel(contact_image, title="RL TacMap Contacts")
        if contact_image is not None and contact_panel is not None:
            contact_panel.update(contact_image, stats)
        if visual_update_due:
            maybe_show_cv(image)
        if int(args_cli.print_every) > 0 and step % int(args_cli.print_every) == 0:
            print(f"[step {step:06d}] {stats}", flush=True)
        print_viewer_camera(step)
        tacmap_reading = focus_tacmap_is_active(env)
        if bool(args_cli.pause_on_tacmap_onset) and tacmap_reading and not had_tacmap_reading:
            pause_for_tacmap_onset(env, panel, image, stats, step)
            if bool(getattr(args_cli, "headless", False)):
                break
        had_tacmap_reading = tacmap_reading
        step += 1
        if args_cli.max_steps > 0 and step >= int(args_cli.max_steps):
            break
        if args_cli.real_time:
            sleep_time = float(env.step_dt) - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    if benchmark_enabled:
        result_path, result = write_benchmark_result(env, env_cfg, benchmark_step_times_s)
        metrics = result["metrics"]
        gpu = result["gpu"]
        peak_reserved = gpu.get("peak_reserved_bytes")
        peak_reserved_gb = 0.0 if peak_reserved is None else float(peak_reserved) / 1.0e9
        print(
            "[BENCHMARK] done "
            f"mean={float(metrics['mean_step_s']) * 1000.0:.3f}ms "
            f"p95={float(metrics['p95_step_s']) * 1000.0:.3f}ms "
            f"sim_hz={float(metrics['sim_hz']):.3f} "
            f"env_steps_s={float(metrics['env_steps_per_s']):.3f} "
            f"real_time={float(metrics['real_time_factor']):.3f}x "
            f"peak_reserved={peak_reserved_gb:.3f}GB output={result_path}",
            flush=True,
        )
        gym_env.close()
        return

    if video_encoder is not None:
        if video_encoder.stdin is not None:
            video_encoder.stdin.close()
        if video_encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed while writing {record_path}")
        print(f"[INFO] Saved tactile observation recording: {record_path}", flush=True)
    if args_cli.save_final_image is not None:
        final_image_path = Path(args_cli.save_final_image).expanduser().resolve()
        final_image_path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(to_numpy_uint8_rgb(image)).save(final_image_path)
        print(f"[INFO] Saved final tactile observation image: {final_image_path}", flush=True)
    gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
