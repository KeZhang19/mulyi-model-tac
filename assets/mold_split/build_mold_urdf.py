#!/usr/bin/env python3
"""Split this two-part mold without healing/fusing/repositioning the source CAD.

CPU only. Requires CadQuery 2.7, NumPy and trimesh. Refuses an existing output
directory. URDF limits are authoring defaults, not measured machine parameters.
"""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyz(values):
    return " ".join(f"{v:.15g}" for v in values)


def shape_record(shape):
    box = shape.BoundingBox()
    return {
        "type": shape.ShapeType(),
        "faces": len(shape.Faces()),
        "valid_brep": shape.isValid(),
        "bbox_mm": [[box.xmin, box.ymin, box.zmin], [box.xmax, box.ymax, box.zmax]],
        "area_mm2": shape.Area(),
        "volume_mm3": shape.Volume() if shape.ShapeType() == "Solid" else None,
        "centroid_mm": list(shape.Center().toTuple()),
    }


def parting_plane(upper, lower):
    """Find the largest opposing, coincident planar face pair; retain CAD axes."""
    candidates = []
    between_centroids = np.subtract(upper.Center().toTuple(), lower.Center().toTuple())
    upper_planes = [(i, f, np.array(f.normalAt().toTuple()), np.array(f.Center().toTuple()))
                    for i, f in enumerate(upper.Faces()) if f.geomType() == "PLANE"]
    for li, lf in enumerate(lower.Faces()):
        if lf.geomType() != "PLANE":
            continue
        n = np.array(lf.normalAt().toTuple())
        if np.dot(n, between_centroids) <= 0:
            continue
        lower_center = np.array(lf.Center().toTuple())
        for ui, uf, upper_normal, upper_center in upper_planes:
            if np.dot(n, upper_normal) > -1 + 1e-10:
                continue
            offset_error = abs(np.dot(n, upper_center - lower_center))
            if offset_error < 1e-6:
                candidates.append((min(lf.Area(), uf.Area()), li, ui, n, offset_error))
    if not candidates:
        raise ValueError("No opposing coincident planar faces; manual inspection required")
    _, li, ui, n, error = max(candidates, key=lambda x: x[0])
    lf, uf = lower.Faces()[li], upper.Faces()[ui]
    common_area = lf.intersect(uf).Area()
    if common_area < 10000:
        raise ValueError("This source does not match the inspected mold's large parting face")
    d = float(np.dot(n, lf.Center().toTuple()))
    return {
        "opening_axis_cad": n.tolist(),
        "plane_equation_mm": "dot(opening_axis_cad, point_mm) = plane_offset_mm",
        "plane_offset_mm": d,
        "closest_plane_point_to_cad_origin_mm": (d * n).tolist(),
        "lower_face_index": li,
        "upper_face_index": ui,
        "coplanarity_error_mm": float(error),
        "common_face_area_mm2": common_area,
        "tilt_from_cad_z_deg": math.degrees(math.acos(float(n[2]))),
    }


def export_mesh(shape, path, linear_mm, angular_rad):
    # CadQuery's tessellate() normally uses relative deflection. Mesh explicitly
    # with isRelative=False first, so the stated tolerance is in millimeters.
    mesher = BRepMesh_IncrementalMesh(shape.wrapped, linear_mm, False, angular_rad, False)
    if not mesher.IsDone():
        raise ValueError("OCCT triangulation failed")
    # Read that triangulation directly: invoking tessellate() can remesh using
    # its relative defaults when OCCT reports a slightly larger face deflection.
    vertices, faces = [], []
    for face in shape.Faces():
        location = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face.wrapped, location)
        if poly is None:
            raise ValueError("A source face was not triangulated")
        offset = len(vertices)
        for i in range(1, poly.NbNodes() + 1):
            point = poly.Node(i).Transformed(location.Transformation())
            vertices.append((point.X(), point.Y(), point.Z()))
        reversed_face = face.wrapped.Orientation() == TopAbs_REVERSED
        for triangle in poly.Triangles():
            indices = [triangle.Value(i) + offset - 1 for i in (1, 2, 3)]
            faces.append(indices[::-1] if reversed_face else indices)
    vertices = np.asarray(vertices, dtype=float) * 0.001
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.export(path, file_type="stl")
    # Read the actual exported float32 STL, weld identical STL corner vertices
    # for validation only, and never rewrite/repair the mesh.
    loaded = trimesh.load_mesh(path, process=True)
    cad_volume = shape.Volume() * 1e-9
    rel_volume_error = abs(loaded.volume - cad_volume) / cad_volume
    finite = bool(np.isfinite(loaded.vertices).all())
    if not finite or rel_volume_error > 0.001:
        raise ValueError(f"Mesh fidelity check failed for {path.name}: {rel_volume_error}")
    _, edge_incidence = np.unique(loaded.edges_sorted, axis=0, return_counts=True)
    if not loaded.is_watertight or not loaded.is_winding_consistent:
        print(f"WARNING {path.name}: surface mesh is not closed/consistent; geometry-only delivery, collision NOT certified", flush=True)
    data = np.concatenate((loaded.triangles, np.repeat(loaded.face_normals[:, None, :], 3, axis=1)), axis=2)
    return {
        "triangles": len(loaded.faces),
        "finite_vertices": finite,
        "watertight": bool(loaded.is_watertight),
        "winding_consistent": bool(loaded.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_incidence == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_incidence > 2)),
        "volume_metric_note": "signed triangle volume; diagnostic only if the mesh is not watertight",
        "bbox_m": loaded.bounds.tolist(),
        "mesh_volume_m3": float(loaded.volume),
        "relative_volume_error": float(rel_volume_error),
        "units": "m",
    }, base64.b64encode(data.astype("<f4").tobytes()).decode("ascii")


def write_urdf(path, plane, stroke, effort, velocity):
    root = ET.Element("robot", name="two_part_hand_mold")
    root.append(ET.Comment("Geometry/kinematics asset. CAD axes preserved, meshes in meters. No measured mass/inertia available."))
    root.append(ET.Comment("Display colors and joint limits are authoring choices, not source CAD material or machine specifications."))
    # Put the child/joint origin on the contact plane, with axes parallel to CAD.
    # The equal and opposite visual origin retains every vertex's closed pose.
    p = np.asarray(plane["closest_plane_point_to_cad_origin_mm"]) * 0.001
    for name, offset, color in (("lower_mold", np.zeros(3), "0.65 0.68 0.72 1"),
                                ("upper_mold", -p, "0.28 0.55 0.82 1")):
        link = ET.SubElement(root, "link", name=name)
        for role in ("visual", "collision"):
            item = ET.SubElement(link, role, name=f"{name}_{role}")
            ET.SubElement(item, "origin", xyz=xyz(offset), rpy="0 0 0")
            geometry = ET.SubElement(item, "geometry")
            ET.SubElement(geometry, "mesh", filename=f"meshes/{name}.stl", scale="1 1 1")
            if role == "visual":
                material = ET.SubElement(item, "material", name=f"{name}_display_only")
                ET.SubElement(material, "color", rgba=color)
    joint = ET.SubElement(root, "joint", name="mold_opening", type="prismatic")
    ET.SubElement(joint, "parent", link="lower_mold")
    ET.SubElement(joint, "child", link="upper_mold")
    ET.SubElement(joint, "origin", xyz=xyz(p), rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz=xyz(plane["opening_axis_cad"]))
    ET.SubElement(joint, "limit", lower="0", upper=str(stroke), effort=str(effort), velocity=str(velocity))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_mold_urdf(args):
    source = args.source.resolve()
    source_hash = sha256(source)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output")
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone or reader.TransferRoots() != 1:
        raise ValueError("Unexpected STEP read result")
    shape = cq.Shape.cast(reader.OneShape())
    solids = shape.Solids()
    if len(solids) != 5 or len(shape.Shells()) != 20 or len(shape.Faces()) != 860:
        raise ValueError("This script is scoped to inspected 1.stp: expected 5 solids, 20 shells, 860 faces")
    main_indices = sorted(range(len(solids)), key=lambda i: solids[i].Volume(), reverse=True)[:2]
    upper_i, lower_i = sorted(main_indices, key=lambda i: solids[i].Center().z, reverse=True)
    upper, lower = solids[upper_i], solids[lower_i]
    plane = parting_plane(upper, lower)
    print("parting_plane", json.dumps(plane), flush=True)
    solid_shells = [sh for so in solids for sh in so.Shells()]
    extra_shells = [sh for sh in shape.Shells() if not any(sh.isSame(known) for known in solid_shells)]
    auxiliary = [so for i, so in enumerate(solids) if i not in main_indices] + extra_shells
    if len(extra_shells) != 15:
        raise ValueError("Unexpected auxiliary shell inventory")
    for directory in (output, output / "source", output / "cad", output / "meshes"):
        directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / "source" / "original.stp")
    mesh_reports, web_meshes, part_reports = {}, {}, {}
    for name, idx in (("lower_mold", lower_i), ("upper_mold", upper_i)):
        part = solids[idx]
        part_reports[name] = {"source_unique_solid_index": idx, **shape_record(part)}
        part.exportBrep(str(output / "cad" / f"{name}.brep"))
        cq.exporters.export(part, str(output / "cad" / f"{name}.step"))
        report, web = export_mesh(part, output / "meshes" / f"{name}.stl", args.linear_mm, args.angular_rad)
        mesh_reports[name], web_meshes[name] = report, web
        print(name, json.dumps(report), flush=True)
    cq.exporters.export(cq.Compound.makeCompound(auxiliary), str(output / "cad" / "unassigned_reference.step"))
    write_urdf(output / "mold.urdf", plane, args.stroke_m, args.effort_n, args.velocity_m_s)
    raw = source.read_text()
    report = {
        "source_sha256": source_hash,
        "source_archive": "source/original.stp",
        "cad_units": "mm",
        "urdf_mesh_units": "m",
        "source_counts": {"solids": len(solids), "shells": len(shape.Shells()), "faces": len(shape.Faces())},
        "source_palette_statements": re.findall(r"^#\d+=(?:COLOUR_RGB|DRAUGHTING_PRE_DEFINED_COLOUR)\(.*?;", raw, re.M | re.S),
        "source_layer_statements": re.findall(r"^#\d+=PRESENTATION_LAYER_ASSIGNMENT\(.*?;", raw, re.M | re.S),
        "source_styled_item_count": raw.count("STYLED_ITEM"),
        "color_note": "Source has a palette but no STYLED_ITEM bindings; neutral/display colors are not source materials.",
        "parts": part_reports,
        "parting_plane": plane,
        "unassigned_reference": {
            "cad": "cad/unassigned_reference.step",
            "reason": "3 isolated tiny solids and 15 auxiliary shells are retained, not arbitrarily attached to either moving link.",
            "geometry": [shape_record(p) for p in auxiliary],
        },
        "tessellation": {"absolute_linear_deflection_mm": args.linear_mm, "angular_deflection_rad": args.angular_rad, "simplification": False},
        "mesh_validation": mesh_reports,
        "joint": {"type": "prismatic", "q_zero": "original CAD closed pose", "stroke_m": args.stroke_m,
                  "effort_n": args.effort_n, "velocity_m_s": args.velocity_m_s,
                  "limits_provenance": "editable authoring defaults, not supplied by source CAD"},
        "physics": {"inertial": "omitted; material/density/mass unknown", "collision": "full surface meshes; topology and importer-specific concave collision configuration still require validation",
                    "collision_ready": all(m["watertight"] and m["winding_consistent"] for m in mesh_reports.values()),
                    "simulator_verified": False},
        "versions": {"cadquery": cq.__version__, "numpy": np.__version__, "trimesh": trimesh.__version__},
    }
    payload = {"meshes": web_meshes, "axis": plane["opening_axis_cad"], "stroke_mm": args.stroke_m * 1000,
               "source_sha256": source_hash}
    template = Path(__file__).with_name("preview_template.html").read_text()
    (output / "preview.html").write_text(template.replace("__MOLD_DATA__", json.dumps(payload)))
    assert sha256(source) == source_hash == sha256(output / "source" / "original.stp")
    report["artifact_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("created", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "1.stp")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("generated"))
    parser.add_argument("--stroke-m", type=float, default=0.15)
    parser.add_argument("--effort-n", type=float, default=100.0)
    parser.add_argument("--velocity-m-s", type=float, default=0.05)
    parser.add_argument("--linear-mm", type=float, default=0.02)
    parser.add_argument("--angular-rad", type=float, default=0.08)
    args = parser.parse_args()
    for name in ("stroke_m", "effort_n", "velocity_m_s", "linear_mm", "angular_rad"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    build_mold_urdf(args)
