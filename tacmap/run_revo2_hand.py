from __future__ import annotations

"""Standalone viewer for the Revo2 right hand and tactile map previews.

This script is intentionally independent from the Sharpawave tactile-align env:
it loads the user's Revo2 hand USD directly, then optionally attaches sampled
tactile map points/normals under each matching touch link.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_USD = str(REPO_ROOT / "assets" / "revo2_system" / "urdf" / "revo2_right.usd")
DEFAULT_URDF = str(REPO_ROOT / "assets" / "revo2_system" / "urdf" / "revo2_right.urdf")
DEFAULT_MAP_DIR = REPO_ROOT / "assets" / "tactilesensor_map"


parser = argparse.ArgumentParser(description="Load and visualize the Revo2 right hand in Isaac Sim.")
parser.add_argument("--usd", type=str, default=DEFAULT_USD, help="Path to the Revo2 right hand USD.")
parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Path to the matching Revo2 right hand URDF.")
parser.add_argument("--prim-path", type=str, default="/World/Revo2RightHand", help="Stage prim path for the hand.")
parser.add_argument("--map-dir", type=str, default=str(DEFAULT_MAP_DIR), help="Directory containing tactile point/normal npy maps.")
parser.add_argument("--hide-tactile-maps", action="store_true", help="Do not draw tactile map points.")
parser.add_argument("--show-normals", action="store_true", help="Draw sampled outward normals.")
parser.add_argument("--point-scale", type=float, default=1.0e-3, help="Scale applied to npy points. Default converts mm to m.")
parser.add_argument("--point-radius", type=float, default=8.0e-4, help="Radius of preview point spheres in meters.")
parser.add_argument("--visual-surface-offset", type=float, default=0.0, help="Visual-only offset along outward normals in meters.")
parser.add_argument("--normal-length", type=float, default=4.0e-3, help="Length of preview normals in meters.")
parser.add_argument("--normal-stride", type=int, default=40, help="Draw one normal every N valid map points.")
parser.add_argument("--preview-stride", type=int, default=1, help="Keep one point every N valid map points before max-point limiting.")
parser.add_argument("--max-points-per-map", type=int, default=15000, help="Maximum preview points per tactile map. Use 0 for no limit.")
parser.add_argument(
    "--color-mode",
    choices=("stripes", "gradient", "red"),
    default="stripes",
    help="Coloring used for tactile map points.",
)
parser.add_argument(
    "--swap-stripe-colors",
    action="store_true",
    help="Swap red row stripes and blue column stripes for visual diagnosis.",
)
parser.add_argument("--stripe-interval", type=int, default=10, help="Row/column spacing for stripe colors.")
parser.add_argument("--print-prim-paths", action="store_true", help="Print matching touch-link prim paths after USD load.")
parser.add_argument("--joint-pos", action="append", default=[], help="Set a joint position, e.g. --joint-pos right_index_proximal_joint=0.3")
parser.add_argument("--pos", nargs=3, type=float, default=(0.0, 0.0, 0.5), help="Hand root position.")
parser.add_argument("--rot", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0), help="Hand root quaternion in wxyz.")
parser.add_argument("--stiffness", type=float, default=3.0, help="Implicit actuator stiffness for Revo2 joints.")
parser.add_argument("--damping", type=float, default=0.1, help="Implicit actuator damping for Revo2 joints.")
parser.add_argument("--effort-limit", type=float, default=2.0, help="Implicit actuator effort limit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: E402


FALLBACK_TOUCH_LINKS = [
    "right_thumb_touch_link",
    "right_index_touch_link",
    "right_middle_touch_link",
    "right_ring_touch_link",
    "right_pinky_touch_link",
]


def parse_urdf_links_and_joints(urdf_path: str) -> tuple[list[str], dict[str, float]]:
    """Return touch links and neutral joint positions from the URDF."""
    if not urdf_path or not os.path.exists(urdf_path):
        return FALLBACK_TOUCH_LINKS, {}

    root = ET.parse(urdf_path).getroot()
    touch_links = [
        link.attrib["name"]
        for link in root.findall("link")
        if "touch_link" in link.attrib.get("name", "")
    ]

    joint_pos: dict[str, float] = {}
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        name = joint.attrib.get("name")
        if not name:
            continue
        lower = 0.0
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib:
            lower = float(limit.attrib["lower"])
        joint_pos[name] = lower

    return touch_links or FALLBACK_TOUCH_LINKS, joint_pos


def parse_joint_overrides(overrides: list[str]) -> dict[str, float]:
    joint_pos = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --joint-pos value: {item!r}. Use name=value.")
        name, value = item.split("=", 1)
        joint_pos[name.strip()] = float(value)
    return joint_pos


def find_descendant_by_name(stage: Usd.Stage, root_path: str, name: str) -> Usd.Prim | None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def make_colors(rows: np.ndarray, cols: np.ndarray, height: int, width: int) -> list[Gf.Vec3f]:
    mode = args_cli.color_mode
    if mode == "red":
        return [Gf.Vec3f(1.0, 0.0, 0.0)] * len(rows)
    if mode == "gradient":
        row_den = max(1, height - 1)
        col_den = max(1, width - 1)
        return [
            Gf.Vec3f(float(col / col_den), float(row / row_den), 1.0 - float(row / row_den))
            for row, col in zip(rows, cols)
        ]

    interval = max(1, int(args_cli.stripe_interval))
    colors = []
    for row, col in zip(rows, cols):
        is_row = int(row) % interval == 0
        is_col = int(col) % interval == 0
        if is_row and is_col:
            colors.append(Gf.Vec3f(1.0, 1.0, 0.0))
        elif is_row and not args_cli.swap_stripe_colors:
            colors.append(Gf.Vec3f(1.0, 0.0, 0.0))
        elif is_row:
            colors.append(Gf.Vec3f(0.0, 0.2, 1.0))
        elif is_col and not args_cli.swap_stripe_colors:
            colors.append(Gf.Vec3f(0.0, 0.2, 1.0))
        elif is_col:
            colors.append(Gf.Vec3f(1.0, 0.0, 0.0))
        else:
            colors.append(Gf.Vec3f(0.86, 0.86, 0.86))
    return colors


def valid_map_samples(points_m: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if points_m.ndim != 3 or points_m.shape[-1] != 3:
        raise ValueError(f"Expected point map with shape (H, W, 3), got {points_m.shape}")
    if normals.shape != points_m.shape:
        raise ValueError(f"Point and normal maps must have the same shape, got {points_m.shape} and {normals.shape}")

    height, width = points_m.shape[:2]
    normal_norm = np.linalg.norm(normals, axis=-1)
    finite = np.isfinite(points_m).all(axis=-1) & np.isfinite(normals).all(axis=-1)
    mask = finite & (normal_norm > 0.5)
    flat_ids = np.flatnonzero(mask.reshape(-1))

    stride = max(1, int(args_cli.preview_stride))
    flat_ids = flat_ids[::stride]
    max_points = int(args_cli.max_points_per_map)
    if max_points > 0 and len(flat_ids) > max_points:
        limit_stride = int(math.ceil(len(flat_ids) / max_points))
        flat_ids = flat_ids[::limit_stride]

    pts = points_m.reshape(-1, 3)[flat_ids]
    nrms = normals.reshape(-1, 3)[flat_ids]
    nrms = nrms / (np.linalg.norm(nrms, axis=-1, keepdims=True) + 1.0e-12)
    rows = flat_ids // width
    cols = flat_ids % width
    return pts, nrms, rows, cols, height, width


def make_stripe_proto_indices(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return prototype ids matching the TacMap debug stripe colors.

    Prototype ids:
    0 = ordinary gray, 1 = red stripe, 2 = blue stripe, 3 = intersection yellow.
    """
    interval = max(1, int(args_cli.stripe_interval))
    is_row = (rows % interval) == 0
    is_col = (cols % interval) == 0
    proto_indices = np.zeros(len(rows), dtype=np.int32)
    if args_cli.swap_stripe_colors:
        proto_indices[is_row] = 2
        proto_indices[is_col] = 1
    else:
        proto_indices[is_row] = 1
        proto_indices[is_col] = 2
    proto_indices[is_row & is_col] = 3
    return proto_indices


def define_colored_point_prototypes(stage: Usd.Stage, proto_root: Sdf.Path) -> list[Sdf.Path]:
    """Define colored sphere prototypes for robust PointInstancer coloring."""
    colors = [
        ("ordinary", Gf.Vec3f(0.86, 0.86, 0.86)),
        ("row_red", Gf.Vec3f(1.0, 0.0, 0.0)),
        ("col_blue", Gf.Vec3f(0.0, 0.2, 1.0)),
        ("cross_yellow", Gf.Vec3f(1.0, 1.0, 0.0)),
    ]
    UsdGeom.Xform.Define(stage, proto_root)
    prototype_paths = []
    for name, color in colors:
        sphere_path = proto_root.AppendPath(name)
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(float(args_cli.point_radius))
        UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([color]))
        prototype_paths.append(sphere_path)
    return prototype_paths


class TactileMapPreview:
    """Draw a tactile map using the live articulation link pose."""

    def __init__(
        self,
        stage: Usd.Stage,
        hand: Articulation,
        body_idx: int,
        link_name: str,
        point_path: Path,
        normal_path: Path,
    ):
        self.hand = hand
        self.body_idx = int(body_idx)
        self.link_name = link_name
        self.device = hand.data.default_joint_pos.device

        points_m = np.load(point_path).astype(np.float32) * float(args_cli.point_scale)
        normals = np.load(normal_path).astype(np.float32)
        all_pts = points_m.reshape(-1, 3)
        all_nrms = normals.reshape(-1, 3)
        pts, nrms, rows, cols, height, width = valid_map_samples(points_m, normals)
        if len(pts) == 0:
            raise ValueError(f"{link_name}: no valid tactile samples in {point_path}")

        self.pts_l = torch.tensor(pts, dtype=torch.float32, device=self.device)
        self.nrms_l = torch.tensor(nrms, dtype=torch.float32, device=self.device)
        self.all_pts_l = torch.tensor(all_pts, dtype=torch.float32, device=self.device)
        self.all_nrms_l = torch.tensor(all_nrms, dtype=torch.float32, device=self.device)
        self.normal_indices = torch.arange(
            0, len(all_pts), max(1, int(args_cli.normal_stride)), dtype=torch.long, device=self.device
        )

        sensor_name = link_name.replace("*", "all").replace(".", "_")
        self.points_path = Sdf.Path(f"/Visuals/TacmapSurfacePoints/env_0/{sensor_name}")
        self.normals_path = Sdf.Path(f"/Visuals/TacmapSurfaceNormals/env_0/{sensor_name}")

        UsdGeom.Xform.Define(stage, self.points_path.GetParentPath())
        self.inst = UsdGeom.PointInstancer.Define(stage, self.points_path)
        self.inst.CreatePositionsAttr()
        self.inst.CreateScalesAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * len(pts)))

        proto_root = self.points_path.AppendPath("Prototypes")
        if args_cli.color_mode == "stripes":
            proto_indices = make_stripe_proto_indices(rows, cols)
            prototype_paths = define_colored_point_prototypes(stage, proto_root)
            self.inst.CreateProtoIndicesAttr(Vt.IntArray(proto_indices.tolist()))
            self.inst.CreatePrototypesRel().SetTargets(prototype_paths)
        else:
            UsdGeom.Xform.Define(stage, proto_root)
            sphere_path = proto_root.AppendPath("sample")
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(float(args_cli.point_radius))
            UsdGeom.Gprim(sphere.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.0, 0.0)]))
            self.inst.CreateProtoIndicesAttr(Vt.IntArray([0] * len(pts)))
            self.inst.CreatePrototypesRel().SetTargets([sphere_path])

            if args_cli.color_mode == "gradient":
                colors = make_colors(rows, cols, height, width)
                primvar = UsdGeom.PrimvarsAPI(self.inst.GetPrim()).CreatePrimvar(
                    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
                )
                primvar.Set(Vt.Vec3fArray(colors))

        self.curves = None
        if args_cli.show_normals:
            UsdGeom.Xform.Define(stage, self.normals_path.GetParentPath())
            self.curves = UsdGeom.BasisCurves.Define(stage, self.normals_path)
            self.curves.CreateTypeAttr("linear")
            self.curves.CreateBasisAttr("bezier")
            self.curves.CreateCurveVertexCountsAttr(Vt.IntArray([2] * int(len(self.normal_indices))))
            self.curves.CreateWidthsAttr(Vt.FloatArray([float(args_cli.point_radius) * 0.35]))
            UsdGeom.Gprim(self.curves.GetPrim()).CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.55, 0.0)]))

        print(
            f"[INFO] {link_name}: prepared {len(pts)} tactile points and "
            f"{len(self.normal_indices)} normal samples from {point_path}"
        )

    def update(self):
        state_w = self.hand.data.body_link_state_w[0, self.body_idx]
        pos_w = state_w[:3]
        quat_w = state_w[3:7]
        q = quat_w.unsqueeze(0).repeat(self.pts_l.shape[0], 1)

        pts_l = self.pts_l
        if abs(float(args_cli.visual_surface_offset)) > 0.0:
            pts_l = pts_l + self.nrms_l * float(args_cli.visual_surface_offset)
        pts_w = quat_apply(q, pts_l) + pos_w
        pts_np = pts_w.detach().cpu().numpy()
        self.inst.GetPositionsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts_np])
        )

        if self.curves is not None:
            idx = self.normal_indices
            n_pts_l = self.all_pts_l[idx]
            if abs(float(args_cli.visual_surface_offset)) > 0.0:
                n_pts_l = n_pts_l + self.all_nrms_l[idx] * float(args_cli.visual_surface_offset)
            n_nrms_l = self.all_nrms_l[idx]
            nq = quat_w.unsqueeze(0).repeat(n_pts_l.shape[0], 1)
            starts_w = quat_apply(nq, n_pts_l) + pos_w
            normals_w = quat_apply(nq, n_nrms_l)
            ends_w = starts_w + normals_w * float(args_cli.normal_length)
            segments = torch.stack([starts_w, ends_w], dim=1).reshape(-1, 3)
            seg_np = segments.detach().cpu().numpy()
            self.curves.GetPointsAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in seg_np])
            )


def draw_available_tactile_maps(stage: Usd.Stage, hand: Articulation, touch_links: list[str]) -> list[TactileMapPreview]:
    map_dir = Path(args_cli.map_dir)
    previews = []
    for link_name in touch_links:
        map_names = [link_name]
        if link_name.endswith("_link"):
            map_names.append(link_name[: -len("_link")])

        point_path = None
        normal_path = None
        for map_name in map_names:
            candidate_point = map_dir / f"{map_name}_point.npy"
            candidate_normal = map_dir / f"{map_name}_normal.npy"
            if candidate_point.exists() and candidate_normal.exists():
                point_path = candidate_point
                normal_path = candidate_normal
                break
        if point_path is None or normal_path is None:
            continue
        if link_name not in hand.body_names:
            print(f"[WARN] Could not find body for {link_name}; skipped tactile preview.")
            continue
        body_idx = hand.body_names.index(link_name)
        try:
            preview = TactileMapPreview(stage, hand, body_idx, link_name, point_path, normal_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue
        preview.update()
        previews.append(preview)
    return previews


def build_hand_cfg(joint_defaults: dict[str, float]) -> ArticulationCfg:
    joint_defaults = dict(joint_defaults)
    joint_defaults.update(parse_joint_overrides(args_cli.joint_pos))
    return ArticulationCfg(
        prim_path=args_cli.prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(args_cli.pos),
            rot=tuple(args_cli.rot),
            joint_pos=joint_defaults,
        ),
        actuators={
            "revo2_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit_sim=float(args_cli.effort_limit),
                stiffness=float(args_cli.stiffness),
                damping=float(args_cli.damping),
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def main():
    if not os.path.exists(args_cli.usd):
        raise FileNotFoundError(f"USD not found: {args_cli.usd}")

    touch_links, joint_defaults = parse_urdf_links_and_joints(args_cli.urdf)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.18, -0.32, 0.68], target=[0.0, 0.0, 0.54])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    hand = Articulation(build_hand_cfg(joint_defaults))
    sim.reset()

    dof_pos = hand.data.default_joint_pos.clone()
    dof_vel = torch.zeros_like(dof_pos)
    hand.write_joint_state_to_sim(dof_pos, dof_vel)
    hand.set_joint_position_target(dof_pos)

    print("[INFO] Revo2 hand loaded.")
    print(f"[INFO] USD : {args_cli.usd}")
    print(f"[INFO] URDF: {args_cli.urdf}")
    print(f"[INFO] articulation joints: {hand.joint_names}")
    print(f"[INFO] articulation bodies: {hand.body_names}")

    stage = sim_utils.get_current_stage()
    if args_cli.print_prim_paths:
        for link_name in touch_links:
            prim = find_descendant_by_name(stage, args_cli.prim_path, link_name)
            print(f"[INFO] touch link {link_name}: {prim.GetPath() if prim else 'NOT FOUND'}")

    tactile_previews = []
    if not args_cli.hide_tactile_maps:
        tactile_previews = draw_available_tactile_maps(stage, hand, touch_links)

    while simulation_app.is_running():
        hand.set_joint_position_target(dof_pos)
        sim.step()
        for preview in tactile_previews:
            preview.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
