#!/usr/bin/env python3
"""Export the actual USD meshes at a calibrated FK pose to GLB and a PNG preview.

CPU only. An optional validator body report uses measured body poses instead of
FK. This is a geometric preview, not a simulated control trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from solve_pregrasp import Kinematics
from validate_assets import load_usd_modules


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT / "assets/bulb/tools"))
from tabletop_validation import body_meshes_local, pose_matrix


def mesh_in_pose(stage, body_path, matrix):
    pieces = []
    for _, vertices, faces in body_meshes_local(stage, body_path):
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.apply_transform(matrix)
        pieces.append(mesh)
    return trimesh.util.concatenate(pieces) if pieces else None


def make_scene(config, body_report=None, down=0.):
    Usd, *_ = load_usd_modules()
    kin = Kinematics()
    robot_root = np.eye(4)
    robot_root[:3, :3] = Rotation.from_euler("z", -np.pi / 2).as_matrix()
    robot_root[:3, 3] = [1., 0., .766]
    joints = dict(config["robot_joint_positions"])
    poses = kin.fk(joints, robot_root)
    peg = config["peg_pose"]
    peg_pose = pose_matrix([*peg["position"], *peg["quaternion"]])
    if body_report:
        if down:
            raise ValueError("--down is only available with FK, without --body-report")
        raw = body_report["root_body_poses"]
        origin = np.asarray(raw["Base_R"][:3]) - robot_root[:3, 3]
        poses = {}
        for name, pose in raw.items():
            poses[name] = pose_matrix(pose)
            poses[name][:3, 3] -= origin
        if body_report.get("history"):
            peg_pose = pose_matrix(body_report["history"][-1]["pose"][0])
            peg_pose[:3, 3] -= origin
    elif down:
        target = poses["base_link"].copy(); target[2, 3] -= down
        names = [f"Joint{i}_R" for i in range(1, 8)]
        seed = np.array([joints[n] for n in names])

        def residual(q):
            actual = kin.fk(joints | dict(zip(names, q)), robot_root)["base_link"]
            return np.r_[(actual[:3, 3] - target[:3, 3]) * 10,
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()]

        fit = least_squares(residual, seed, bounds=kin.bounds[:7].T, max_nfev=250)
        if np.linalg.norm(fit.fun) > .001:
            raise ValueError("Could not find a pose for the requested vertical displacement")
        poses = kin.fk(joints | dict(zip(names, fit.x)), robot_root)
        peg_pose[2, 3] -= down
    stage = Usd.Stage.Open(str(ROOT / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right_visual.usd"))
    root = str(stage.GetDefaultPrim().GetPath())
    scene = trimesh.Scene()
    for name, pose in poses.items():
        if not stage.GetPrimAtPath(root + "/" + name).IsValid():
            continue
        mesh = mesh_in_pose(stage, root + "/" + name, pose)
        if mesh is None:
            continue
        color = [192, 202, 214, 255] if not name.startswith("right_") else [122, 142, 159, 255]
        if "rubber" in name:
            color = [69, 81, 93, 255]
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    for name, pose, color in (("peg", peg_pose, [226, 171, 46, 255]),
                              ("socket", pose_matrix([.45, .1, .76, 1, 0, 0, 0]), [63, 100, 132, 255])):
        stage = Usd.Stage.Open(str(PACKAGE / f"{name}.usd"))
        # Select the complete canonical visible mesh explicitly: neither the
        # hidden sensing surface nor the visual direction marker is duplicated.
        root = str(stage.GetDefaultPrim().GetPath())
        mesh = mesh_in_pose(stage, root + "/VisualMesh", pose)
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    table = trimesh.creation.box(extents=(1.2, 1.6, .76))
    table.apply_translation([.55, 0., .38]); table.visual.vertex_colors = [192, 183, 170, 255]
    scene.add_geometry(table, node_name="table", geom_name="table")
    return scene


def save_png(scene, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 8), facecolor="#f1f4f7")
    full = fig.add_axes((.01, .08, .53, .82), projection="3d")
    close = fig.add_axes((.55, .11, .43, .76), projection="3d")
    light = np.array([-.4, -.7, 1.0]); light /= np.linalg.norm(light)
    for ax in (full, close):
        triangles, colors = [], []
        for name, mesh in scene.geometry.items():
            if ax == close and name not in ("peg", "socket", "base_link", "table") and not name.startswith("right_"):
                continue
            if name == "table":
                # Plot the tabletop within each view. The exported GLB keeps
                # the complete original table. Matplotlib does not clip 3D
                # polygons to the axes, so full table sides obscure the inset.
                xmin, xmax, ymin, ymax = ((-.05, 1.15, -.32, .48) if ax == full else (.37, .65, .015, .25))
                vertices = np.array([[xmin, ymin, .76], [xmax, ymin, .76],
                                     [xmax, ymax, .76], [xmin, ymax, .76]])
                triangles.append(vertices[[[0, 1, 2], [0, 2, 3]]])
                colors.append(np.tile(np.array([192, 183, 170]) / 255, (2, 1)))
                continue
            faces = mesh.faces
            # Preserve the actual surface. Grid contours are avoided because
            # exported robot CAD has many small tessellation edges.
            shade = .40 + .60 * np.maximum(mesh.face_normals @ light, 0)
            color = np.asarray(mesh.visual.vertex_colors[0, :3], dtype=float) / 255
            triangles.append(mesh.vertices[faces])
            colors.append(color * shade[:, None])
        # One collection sorts faces across bodies, avoiding whole-body
        # painter ordering that can draw the table over the robot.
        ax.add_collection3d(Poly3DCollection(np.concatenate(triangles), facecolors=np.concatenate(colors),
                                            linewidths=0, edgecolors="none", antialiased=False))
    full.set(xlim=(-.05, 1.18), ylim=(-.32, .48), zlim=(.70, 1.45), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    full.set_box_aspect((1.23, .8, .75)); full.view_init(elev=23, azim=-68)
    close.set(xlim=(.37, .65), ylim=(.015, .25), zlim=(.75, 1.025), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    close.set_box_aspect((.28, .235, .275)); close.view_init(elev=23, azim=-60)
    full.set_title("Tianji arm / Revo3 hand / original tabletop", pad=14)
    close.set_title("D-peg grip and open socket", pad=14)
    fig.suptitle(label, fontsize=15)
    fig.text(.02, .015, "Geometry preview from checked-in CAD/USD at calibrated FK poses; units: metres", fontsize=10, color="#435363")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE / "pregrasp.json")
    parser.add_argument("--body-report", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=PACKAGE / "scene_preview")
    parser.add_argument("--down", type=float, default=0., help="Optional FK-only vertical displacement in metres")
    parser.add_argument("--skip-png", action="store_true")
    args = parser.parse_args()
    scene = make_scene(json.loads(args.config.read_text()),
                       json.loads(args.body_report.read_text()) if args.body_report else None, args.down)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    glb = args.output_prefix.with_suffix(".glb")
    scene.export(str(glb))
    png = args.output_prefix.with_suffix(".png")
    if not args.skip_png:
        save_png(scene, png, f"D-peg insertion — {args.config.name}; downward displacement {args.down * 1000:.0f} mm")
    print(json.dumps({"meshes": len(scene.geometry), "glb": str(glb),
                      "png": None if args.skip_png else str(png), "bounds_m": scene.bounds.tolist()}))


if __name__ == "__main__":
    main()
