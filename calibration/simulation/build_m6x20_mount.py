#!/usr/bin/env python3
"""Build a shortened V5 calibration mount for M6x20 breadboard screws.

The V4 screw bearing plane is 24 mm above its bottom face.  Removing the
lowest 10 mm and re-basing the part puts that plane 14 mm above the optical
table, leaving 6 mm of nominal M6x20 projection (5 mm with a 1 mm washer).
The existing 6.8 mm through holes and 12 mm counterbores are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "calibration/assets/revo3_breadboard_mount/generated_calibration_v4"
TOUCH_HAND = ROOT / "assets/revo21_right_touch"
DEFAULT_OUTPUT = ROOT / "calibration/assets/revo3_breadboard_mount/generated_calibration_v5_m6x20"

CUT_HEIGHT_MM = 10.0
SCREW_LENGTH_MM = 20.0
BEARING_HEIGHT_V4_MM = 24.0
BEARING_HEIGHT_V5_MM = BEARING_HEIGHT_V4_MM - CUT_HEIGHT_MM
NOMINAL_PROJECTION_MM = SCREW_LENGTH_MM - BEARING_HEIGHT_V5_MM
MOUNT_HOLES_MM = ((18.0, 235.0), (93.0, 235.0), (143.0, 10.0), (193.0, 10.0))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shifted_origin(origin: ET.Element, dz_m: float) -> None:
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    xyz[2] += dz_m
    origin.set("xyz", " ".join(f"{v:.12g}" for v in xyz))


def connected_components(mesh: trimesh.Trimesh) -> int:
    parent = np.arange(len(mesh.vertices), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for a, b, c in mesh.faces:
        union(int(a), int(b))
        union(int(b), int(c))
    used = np.unique(mesh.faces)
    return len({find(int(v)) for v in used})


def repair_occ_tessellation(shape: cq.Shape, path: Path) -> tuple[trimesh.Trimesh, list[dict]]:
    """Remove the inherited pole degeneracy and close two sampled BSpline gaps."""
    mesh = trimesh.load(path, force="mesh", process=True)
    faces = mesh.faces[mesh.area_faces > 1e-12]
    signatures = np.sort(faces, axis=1)
    _, unique = np.unique(signatures, axis=0, return_index=True)
    mesh = trimesh.Trimesh(mesh.vertices, faces[np.sort(unique)], process=True)
    counts = np.bincount(mesh.edges_unique_inverse)
    if np.any(counts > 2):
        raise RuntimeError("Unexpected non-manifold edges remain after removing degenerate faces")
    boundary = mesh.edges_unique[counts == 1]
    neighbors: dict[int, list[int]] = {}
    for left, right in boundary:
        neighbors.setdefault(int(left), []).append(int(right))
        neighbors.setdefault(int(right), []).append(int(left))
    if not all(len(values) == 2 for values in neighbors.values()):
        raise RuntimeError("Tessellation boundary is not a collection of closed loops")

    pending = set(neighbors)
    loops = []
    while pending:
        first = min(pending)
        loop, previous, current = [first], None, first
        while True:
            following = next(value for value in neighbors[current] if value != previous)
            if following == first:
                break
            loop.append(following)
            previous, current = current, following
        pending.difference_update(loop)
        loops.append(loop)
    if sorted(len(loop) for loop in loops) != [4, 4]:
        raise RuntimeError(f"Unexpected inherited tessellation gaps: {[len(loop) for loop in loops]}")

    directed_edges = {tuple(edge) for edge in mesh.edges.tolist()}
    additions = []
    reports = []
    for loop in loops:
        candidates = (
            ([loop[0], loop[1], loop[2]], [loop[0], loop[2], loop[3]]),
            ([loop[0], loop[1], loop[3]], [loop[1], loop[2], loop[3]]),
        )
        candidate_errors = []
        for candidate in candidates:
            maximum = 0.0
            for triangle in candidate:
                xyz = mesh.vertices[list(triangle)]
                for i in range(9):
                    for j in range(9 - i):
                        point = xyz[0] * (1 - (i + j) / 8) + xyz[1] * i / 8 + xyz[2] * j / 8
                        maximum = max(maximum, shape.distance(cq.Vertex.makeVertex(*point)))
            candidate_errors.append(maximum)
        selected = [list(triangle) for triangle in candidates[int(np.argmin(candidate_errors))]]
        reverse = None
        for triangle in selected:
            for left, right in zip(triangle, np.roll(triangle, -1)):
                if (int(left), int(right)) in directed_edges:
                    reverse = True
                    break
                if (int(right), int(left)) in directed_edges:
                    reverse = False
                    break
            if reverse is not None:
                break
        if reverse is None:
            raise RuntimeError("Could not orient tessellation recovery patch")
        if reverse:
            selected = [triangle[::-1] for triangle in selected]
        additions.extend(selected)
        reports.append(
            {
                "boundary_vertices": len(loop),
                "triangles_added": len(selected),
                "max_sampled_cad_surface_distance_mm": min(candidate_errors),
                "acceptance_tolerance_mm": 0.02,
            }
        )

    mesh = trimesh.Trimesh(mesh.vertices, np.vstack((mesh.faces, additions)), process=True)
    counts = np.bincount(mesh.edges_unique_inverse)
    if not mesh.is_watertight or not mesh.is_winding_consistent or np.any(counts != 2):
        raise RuntimeError("Recovered STL is not a closed consistently wound mesh")
    if max(report["max_sampled_cad_surface_distance_mm"] for report in reports) >= 0.02:
        raise RuntimeError("Recovered patch exceeds the configured CAD mesh tolerance")
    mesh.export(path)
    return trimesh.load(path, force="mesh", process=True), reports


def write_3mf(mesh: trimesh.Trimesh, path: Path) -> None:
    vertices = "".join(
        f'<vertex x="{x:.9g}" y="{y:.9g}" z="{z:.9g}"/>' for x, y, z in mesh.vertices
    )
    triangles = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" xml:lang="zh-CN" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        '<resources><object id="1" type="model"><mesh><vertices>'
        + vertices
        + '</vertices><triangles>'
        + triangles
        + '</triangles></mesh></object></resources><build><item objectid="1"/></build></model>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '</Types>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        '</Relationships>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("3D/3dmodel.model", model)


def write_preview(mesh: trimesh.Trimesh, path: Path) -> None:
    triangles = mesh.triangles
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    intensity = 0.38 + 0.62 * np.clip(mesh.face_normals @ light, 0.0, 1.0)
    base = np.array([0.25, 0.56, 0.52])
    colors = np.c_[np.clip(base[None, :] * intensity[:, None], 0.0, 1.0), np.ones(len(intensity))]
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles, facecolors=colors, edgecolor="none", linewidth=0.0, alpha=1.0
    )
    ax.add_collection3d(collection)
    lo, hi = mesh.bounds
    margin = np.maximum((hi - lo) * 0.03, 0.8)
    ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
    ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
    ax.set_zlim(lo[2] - margin[2], hi[2] + margin[2])
    ax.view_init(elev=27, azim=-53)
    ax.set_xlabel("X / mm")
    ax.set_ylabel("Y / mm")
    ax.set_zlabel("Z / mm")
    ax.set_title("Revo3 calibration lower mount V5 — M6×20, bottom shortened 10 mm")
    ax.set_box_aspect(hi - lo)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_mount_urdf(output: Path, mesh_name: str) -> Path:
    source = ET.parse(V4 / "urdf/lower_calibration_v4.urdf").getroot()
    source.set("name", "calibration_lower_mount_v5_m6x20")
    for mesh in source.iter("mesh"):
        mesh.set("filename", f"../meshes/mount/{mesh_name}")
    for joint in source.findall("joint"):
        shifted_origin(joint.find("origin"), -CUT_HEIGHT_MM / 1000.0)
    ET.indent(source, space="  ")
    target = output / "urdf/lower_calibration_v5_m6x20.urdf"
    ET.ElementTree(source).write(target, encoding="utf-8", xml_declaration=True)
    return target


def write_touch_assembly(output: Path, fixture_path: Path) -> tuple[Path, dict[str, str]]:
    hand_path = TOUCH_HAND / "urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf"
    fixture = ET.parse(fixture_path).getroot()
    hand = ET.parse(hand_path).getroot()
    old_combo = ET.parse(V4 / "urdf/revo3_right_calibration_lower_mount_v4.urdf").getroot()
    mount = copy.deepcopy(old_combo.find("joint[@name='lower_mount_to_right_hand']"))
    mount.find("child").set("link", "base_link")
    shifted_origin(mount.find("origin"), -CUT_HEIGHT_MM / 1000.0)

    assembled = ET.Element("robot", name="revo21_touch_calibration_lower_mount_v5_m6x20")
    for element in fixture:
        assembled.append(copy.deepcopy(element))

    hashes: dict[str, str] = {}
    for element in hand:
        copied = copy.deepcopy(element)
        for mesh in copied.iter("mesh"):
            original = (hand_path.parent / mesh.get("filename")).resolve()
            relative = original.relative_to(TOUCH_HAND)
            destination = output / "meshes/hand" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(original, destination)
            mesh.set("filename", Path(os.path.relpath(destination, output / "urdf")).as_posix())
            hashes[str(destination.relative_to(output))] = sha256(destination)
        assembled.append(copied)
    assembled.append(mount)
    ET.indent(assembled, space="  ")
    target = output / "urdf/revo21_touch_calibration_lower_mount_v5_m6x20.urdf"
    ET.ElementTree(assembled).write(target, encoding="utf-8", xml_declaration=True)

    shutil.copytree(TOUCH_HAND / "marker_positions", output / "reference/marker_positions")
    shutil.copytree(TOUCH_HAND / "pressure_taxels", output / "reference/pressure_taxels")
    return target, hashes


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "cad").mkdir()
    (output / "meshes/mount").mkdir(parents=True)
    (output / "urdf").mkdir()

    source_brep = V4 / "cad/lower_calibration_v4.brep"
    source = cq.importers.importBrep(str(source_brep)).val()
    # Keep all material at and above the new contact plane, then re-base that
    # plane to Z=0 so URDF Z=0 remains the optical-table mounting plane.
    keep = cq.Solid.makeBox(240.0, 270.0, 40.0, cq.Vector(-10.0, -10.0, CUT_HEIGHT_MM))
    mount = source.intersect(keep).clean().translate((0.0, 0.0, -CUT_HEIGHT_MM))
    if not mount.isValid() or len(mount.Solids()) != 1:
        raise RuntimeError("Shortened mount is not one valid CAD solid")

    brep_path = output / "cad/lower_calibration_v5_m6x20.brep"
    step_path = output / "cad/lower_calibration_v5_m6x20.step"
    hardware_path = output / "cad/reference_m6x20_envelopes.step"
    stl_path = output / "meshes/mount/lower_calibration_v5_m6x20_print.stl"
    mf3_path = output / "meshes/mount/lower_calibration_v5_m6x20_print.3mf"
    mount.exportBrep(str(brep_path))
    cq.exporters.export(mount, str(step_path))
    cq.exporters.export(mount, str(stl_path), tolerance=0.02, angularTolerance=0.05)
    mesh, recovery = repair_occ_tessellation(mount, stl_path)
    write_3mf(mesh, mf3_path)
    cad_readback = cq.importers.importBrep(str(brep_path)).val()

    screws = []
    for x, y in MOUNT_HOLES_MM:
        shank = cq.Solid.makeCylinder(3.0, SCREW_LENGTH_MM, cq.Vector(x, y, -NOMINAL_PROJECTION_MM))
        head = cq.Solid.makeCylinder(5.0, 6.0, cq.Vector(x, y, BEARING_HEIGHT_V5_MM))
        screws.append(shank.fuse(head))
    cq.exporters.export(cq.Compound.makeCompound(screws), str(hardware_path))

    fixture_path = write_mount_urdf(output, stl_path.name)
    assembly_path, hand_hashes = write_touch_assembly(output, fixture_path)

    step_solids = cq.importers.importStep(str(step_path)).solids().vals()
    bounds = cad_readback.BoundingBox()
    source_volume = source.Volume()
    expected_volume = cad_readback.Volume()
    step_volume = sum(s.Volume() for s in step_solids)
    components = connected_components(mesh)
    hardware_intersections = [cad_readback.intersect(s).Volume() for s in screws]

    hole_checks = []
    for index, (x, y) in enumerate(MOUNT_HOLES_MM, start=1):
        checks = []
        for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            dx, dy = math.cos(theta), math.sin(theta)
            for radius, z, expected_inside in (
                (3.35, 0.01, False),
                (3.35, BEARING_HEIGHT_V5_MM - 0.01, False),
                (3.45, BEARING_HEIGHT_V5_MM - 0.01, True),
                (5.95, BEARING_HEIGHT_V5_MM + 0.01, False),
                (6.05, BEARING_HEIGHT_V5_MM + 0.01, True),
            ):
                actual = cad_readback.isInside(cq.Vector(x + radius * dx, y + radius * dy, z), 1e-7)
                if actual != expected_inside:
                    raise AssertionError((index, theta, radius, z, expected_inside, actual))
                checks.append(True)
        hole_checks.append({"id": f"M{index}", "xy_mm": [x, y], "probes_passed": len(checks)})

    with zipfile.ZipFile(mf3_path) as archive:
        model_name = next(name for name in archive.namelist() if name.endswith(".model"))
        model_text = archive.read(model_name).decode("utf-8")
        unit_is_mm = 'unit="millimeter"' in model_text or "unit='millimeter'" in model_text

    referenced_meshes = []
    for urdf in (fixture_path, assembly_path):
        root = ET.parse(urdf).getroot()
        for spec in root.iter("mesh"):
            path = (urdf.parent / spec.get("filename")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            referenced_meshes.append(path)

    validation = {
        "schema": "revo3_calibration_mount_validation_v5_m6x20",
        "cad": {
            "valid": cad_readback.isValid(),
            "solids": len(cad_readback.Solids()),
            "volume_mm3": expected_volume,
            "source_v4_volume_mm3": source_volume,
            "removed_volume_mm3": source_volume - expected_volume,
            "bounds_mm": [[bounds.xmin, bounds.ymin, bounds.zmin], [bounds.xmax, bounds.ymax, bounds.zmax]],
            "step_readback_valid": len(step_solids) == 1 and step_solids[0].isValid(),
            "step_solids": len(step_solids),
            "step_volume_relative_error": abs(step_volume - expected_volume) / expected_volume,
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "triangles": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "connected_components": components,
            "bounds_mm": mesh.bounds.tolist(),
            "cad_volume_relative_error": abs(float(mesh.volume) - expected_volume) / expected_volume,
            "3mf_unit_is_millimeter": unit_is_mm,
            "recovered_source_patches": recovery,
        },
        "fastener": {
            "type": "M6 socket-head cap screw envelope (unthreaded reference)",
            "length_under_head_mm": SCREW_LENGTH_MM,
            "existing_through_diameter_mm": 6.8,
            "existing_counterbore_diameter_mm": 12.0,
            "bearing_plane_above_new_bottom_mm": BEARING_HEIGHT_V5_MM,
            "nominal_projection_without_washer_mm": NOMINAL_PROJECTION_MM,
            "nominal_projection_with_1mm_washer_mm": NOMINAL_PROJECTION_MM - 1.0,
            "actual_optical_table_thread_engagement_verified": False,
            "hardware_intersection_mm3": hardware_intersections,
            "hole_checks": hole_checks,
        },
        "urdf": {
            "fixture": str(fixture_path.relative_to(output)),
            "touch_assembly": str(assembly_path.relative_to(output)),
            "mesh_references": len(referenced_meshes),
            "missing_mesh_references": 0,
            "root_frame_is_new_bottom_plane": True,
            "all_fixture_and_hand_mount_z_frames_shifted_m": -CUT_HEIGHT_MM / 1000.0,
        },
        "physical_validation": False,
        "assembly_status": "REVIEW_REQUIRED",
    }
    required = (
        cad_readback.isValid(),
        len(cad_readback.Solids()) == 1,
        len(step_solids) == 1,
        step_solids[0].isValid(),
        mesh.is_watertight,
        mesh.is_winding_consistent,
        components == 1,
        unit_is_mm,
        max(hardware_intersections) < 1e-6,
        abs(float(mesh.bounds[0, 2])) < 1e-6,
        abs(float(mesh.bounds[1, 2]) - 35.0) < 1e-6,
        NOMINAL_PROJECTION_MM >= 5.0,
    )
    if not all(required):
        raise RuntimeError(f"Validation failed: {required}")

    design = {
        "schema": "revo3_calibration_mount_design_v5_m6x20",
        "units": "millimeter; URDF uses meter",
        "source_v4_brep": str(source_brep.relative_to(ROOT)),
        "source_v4_sha256": sha256(source_brep),
        "modification": "Remove the lowest 10 mm, preserve all higher geometry, then re-base the new bottom to Z=0.",
        "cut_height_mm": CUT_HEIGHT_MM,
        "outer_bounds_mm": [[0.0, -3.0, 0.0], [220.0, 249.0, 35.0]],
        "mount_holes_mm": [list(v) for v in MOUNT_HOLES_MM],
        "fastener": validation["fastener"],
        "frame_shift_from_v4_m": [0.0, 0.0, -CUT_HEIGHT_MM / 1000.0],
        "unchanged": [
            "M6 hole XY positions",
            "6.8 mm through-hole diameter",
            "12 mm counterbore diameter",
            "wrist pocket and supports above the cut plane",
            "40 mm spherical calibration pocket",
            "55 mm AprilTag pocket",
            "touch-hand geometry, joints, limits, inertias, and sensor frames",
        ],
        "limitations": [
            "No physical print or optical-table assembly test was performed.",
            "The optical-table entrance chamfer, blind-hole depth, and usable thread depth were not measured.",
            "Nominal projection is not proof of actual engaged thread length.",
        ],
    }
    manifest_files = [brep_path, step_path, hardware_path, stl_path, mf3_path, fixture_path, assembly_path]
    manifest = {
        "schema": "revo3_calibration_mount_manifest_v5_m6x20",
        "files_sha256": {str(p.relative_to(output)): sha256(p) for p in manifest_files},
        "hand_mesh_sha256": hand_hashes,
    }
    (output / "design.json").write_text(json.dumps(design, indent=2, ensure_ascii=False) + "\n")
    (output / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_preview(mesh, output / "preview.png")
    (output / "README.md").write_text(
        "# Revo3 标定下支架 V5（M6×20）\n\n"
        "本版从 V4 底面裁去 10 mm，并把新底面重新定义为 Z=0；V4 原文件不覆盖。\n"
        "四个固定孔仍为 Ø6.8 mm 通孔、Ø12 mm 平底藏头孔，孔位不变。\n\n"
        "M6 螺丝承靠面距新底面 14 mm，因此 M6×20 在不加垫片时名义伸出 6 mm；\n"
        "加 1 mm 垫片时名义伸出 5 mm。这里的伸出量不等于已经实测的有效啮合量，\n"
        "光学平台入口倒角、盲孔深度和有效牙深仍须实装确认。\n\n"
        "打印文件：`meshes/mount/lower_calibration_v5_m6x20_print.3mf`；\n"
        "CAD：`cad/lower_calibration_v5_m6x20.step`；独立支架 URDF：\n"
        "`urdf/lower_calibration_v5_m6x20.urdf`；当前视触觉手组合 URDF：\n"
        "`urdf/revo21_touch_calibration_lower_mount_v5_m6x20.urdf`。\n\n"
        "支架、手和三个标定辅助 frame 相对光学平台均比 V4 下移 10 mm；X/Y 和全部手部关节定义不变。\n"
        "`validation.json` 记录 CAD、STEP、STL、孔径、螺丝包络和 URDF 网格检查。未做实物装配或承载验收。\n"
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output.resolve())
