#!/usr/bin/env python3
"""Build the original, printable D-peg/socket geometry without launching Isaac Sim.

Run with a Python environment containing CadQuery, numpy, trimesh and matplotlib.
CAD/STEP/STL use millimetres; the self-contained USD assets use metres and Z-up.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh


PACKAGE = Path(__file__).resolve().parents[1]
SHAFT_RADIUS = 12.0
SHAFT_FLAT = 6.0
SHAFT_LENGTH = 35.0
TIP_CHAMFER = 1.0
GRIP_RADIUS = 25.0
GRIP_HEIGHT = 50.0
CLEARANCE = 0.75
SOCKET_WIDTH = 80.0
SOCKET_HEIGHT = 50.0
HOLE_DEPTH = 30.0
ENTRY_CHAMFER = 1.0
TESSELLATION_TOLERANCE_MM = 0.03


def d_solid(radius, flat_x, height, z=0.0):
    """Extrude the convex section x*x+y*y <= radius**2 and x <= flat_x."""
    cylinder = cq.Workplane("XY").circle(radius).extrude(height)
    half_space = cq.Workplane("XY").box(
        2 * radius, 2 * radius, height, centered=(False, True, False)
    ).translate((flat_x - 2 * radius, 0, 0))
    return cylinder.intersect(half_space).translate((0, 0, z))


def mesh_of(shape):
    vertices, faces = shape.val().tessellate(TESSELLATION_TOLERANCE_MM, 0.08)
    mesh = trimesh.Trimesh(
        vertices=np.array([v.toTuple() for v in vertices]), faces=np.asarray(faces), process=True
    )
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def radial_profile(theta, radius, flat):
    direction = np.array([math.cos(theta), math.sin(theta)])
    distance = radius
    if direction[0] > 0:
        distance = min(distance, flat / direction[0])
    return direction * distance


def convex_cell_mesh(vertices):
    """Triangulate the exact convex hull of a small cell, without SciPy.

    Supporting planes are inexpensive to enumerate for the twelve cell vertices.
    Coplanar points are reduced to a 2-D hull before triangulating each face.
    """
    vertices = np.asarray(vertices, dtype=float)
    planes = {}
    for i, j, k in itertools.combinations(range(len(vertices)), 3):
        normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        distances = (vertices - vertices[i]) @ normal
        if distances.max() <= 1e-7:
            pass
        elif distances.min() >= -1e-7:
            normal = -normal
        else:
            continue
        ids = tuple(np.flatnonzero(np.abs(distances) < 1e-7).tolist())
        planes[ids] = normal
    triangles = []
    for ids, normal in planes.items():
        origin = vertices[ids[0]]
        u = vertices[ids[1]] - origin
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        coordinates = {i: ((vertices[i] - origin) @ u, (vertices[i] - origin) @ v) for i in ids}
        ordered = sorted(ids, key=lambda i: coordinates[i])

        def cross(a, b, c):
            aa, bb, cc = (np.asarray(coordinates[i]) for i in (a, b, c))
            return (bb[0] - aa[0]) * (cc[1] - aa[1]) - (bb[1] - aa[1]) * (cc[0] - aa[0])

        lower, upper = [], []
        for chain, points in ((lower, ordered), (upper, reversed(ordered))):
            for index in points:
                while len(chain) >= 2 and cross(chain[-2], chain[-1], index) <= 1e-7:
                    chain.pop()
                chain.append(index)
        polygon = lower[:-1] + upper[:-1]
        triangles.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, len(polygon) - 1))
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)
    mesh.remove_unreferenced_vertices()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def socket_wall_colliders():
    """Explicit convex radial wall cells; never hull the entire concave socket.

    Include the square corners and both D-profile junctions in the partition.
    The final millimetre uses a bevelled inner wall matching the CAD entrance.
    """
    radius, flat = SHAFT_RADIUS + CLEARANCE, SHAFT_FLAT + CLEARANCE
    angles = list(np.linspace(0, 2 * math.pi, 129)[:-1])
    angles += [math.pi / 4 + k * math.pi / 2 for k in range(4)]
    for r, f in ((radius, flat), (radius + ENTRY_CHAMFER, flat + ENTRY_CHAMFER)):
        angle = math.acos(f / r)
        angles.extend((angle, 2 * math.pi - angle))
    angles = sorted(set(round(a, 12) for a in angles))
    levels = (SOCKET_HEIGHT - HOLE_DEPTH, SOCKET_HEIGHT - ENTRY_CHAMFER, SOCKET_HEIGHT)
    walls = []
    for index, a in enumerate(angles):
        b = angles[(index + 1) % len(angles)]
        corners = []
        for z in levels:
            bevel = max(0.0, z - (SOCKET_HEIGHT - ENTRY_CHAMFER))
            inner_a = radial_profile(a, radius + bevel, flat + bevel)
            inner_b = radial_profile(b, radius + bevel, flat + bevel)
            outer = []
            for theta in (a, b):
                direction = np.array([math.cos(theta), math.sin(theta)])
                outer.append(direction * (SOCKET_WIDTH / 2 / np.abs(direction).max()))
            corners.extend((*xy, z) for xy in (inner_a, outer[0], outer[1], inner_b))
        mesh = convex_cell_mesh(corners)
        walls.append((f"Wall_{index:03d}", mesh))
    return walls


def fmt(value):
    return format(float(value), ".10g")


def tuple_array(values):
    return "[" + ", ".join("(" + ", ".join(fmt(x) for x in row) + ")" for row in values) + "]"


def usd_mesh(name, mesh, color, *, collision=False, material="", invisible=False):
    vertices = np.asarray(mesh.vertices) * 0.001
    faces = np.asarray(mesh.faces)
    api = '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]'
    lines = [f'        def Mesh "{name}"' + (f" (prepend apiSchemas = {api})" if collision else ""), "        {"]
    lines += [
        f"            point3f[] points = {tuple_array(vertices)}",
        f"            int[] faceVertexCounts = [{', '.join(['3'] * len(faces))}]",
        f"            int[] faceVertexIndices = [{', '.join(str(int(i)) for i in faces.ravel())}]",
        f"            float3[] extent = {tuple_array([vertices.min(axis=0), vertices.max(axis=0)])}",
        '            uniform token subdivisionScheme = "none"',
        '            uniform token orientation = "rightHanded"',
        f"            color3f[] primvars:displayColor = {tuple_array([color])}",
    ]
    if invisible:
        lines.append('            token visibility = "invisible"')
    if collision:
        lines += [
            "            bool physics:collisionEnabled = true",
            '            uniform token physics:approximation = "convexHull"',
            "            float physxCollision:contactOffset = 0.0001",
            "            float physxCollision:restOffset = 0",
            f"            rel material:binding:physics = <{material}>",
        ]
    lines.append("        }")
    return "\n".join(lines)


def material_text(root, name, static, dynamic):
    return f'''        def Material "{name}" (prepend apiSchemas = ["PhysicsMaterialAPI"])
        {{
            float physics:staticFriction = {static}
            float physics:dynamicFriction = {dynamic}
            float physics:restitution = 0
        }}'''


def write_usd(path, root, visual, collisions, *, kinematic, mass):
    color = (0.90, 0.65, 0.17) if not kinematic else (0.21, 0.32, 0.43)
    lines = [f'''#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI", "PhysxContactReportAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    bool physics:kinematicEnabled = {str(kinematic).lower()}
    float physics:mass = {mass}
    bool physxRigidBody:disableGravity = {str(kinematic).lower()}
    float physxRigidBody:maxDepenetrationVelocity = 0.2
    uint physxRigidBody:solverPositionIterationCount = 32
    uint physxRigidBody:solverVelocityIterationCount = 8
    float physxContactReport:threshold = 0
    def Scope "Materials"
    {{''', material_text(root, "GripContact", 0.9, 0.8), material_text(root, "InsertionContact", 0.5, 0.4), "    }"]
    lines.append(usd_mesh("VisualMesh", visual, color))
    lines.append(usd_mesh("TactileSurface", visual, color, invisible=True))
    if not kinematic:
        # Paint-like arrow identifying the outward normal of the flat (+X).
        marker = trimesh.Trimesh(
            vertices=[(-9, -2, 85.02), (7, -2, 85.02), (7, -5, 85.02),
                      (14, 0, 85.02), (7, 5, 85.02), (7, 2, 85.02), (-9, 2, 85.02)],
            faces=[(0, 1, 5), (0, 5, 6), (2, 3, 4)], process=False,
        )
        lines.append(usd_mesh("FlatDirectionMarker", marker, (0.12, 0.21, 0.36)))
    lines.append('    def Scope "Collisions"\n    {')
    for name, mesh in collisions:
        mat = "GripContact" if name == "Grip" else "InsertionContact"
        lines.append(usd_mesh(name, mesh, color, collision=True, material=f"/{root}/Materials/{mat}", invisible=True))
    lines.extend(("    }", "}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def fingerprint(mesh):
    return {
        "vertices": len(mesh.vertices), "triangles": len(mesh.faces),
        "bounds_m": (mesh.bounds * 0.001).tolist(),
        "mesh_sha256": hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest(),
        "watertight": bool(mesh.is_watertight), "volume_m3": float(mesh.volume * 1e-9),
    }


def preview(peg, socket):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 8), facecolor="#f2f4f6")
    ax = fig.add_axes((.02, .12, .52, .76), projection="3d")
    from matplotlib.colors import to_rgb
    for mesh, color, shift in ((socket, "#526f89", (-50, 0, 0)), (peg, "#e8b03c", (50, 0, 0))):
        faces = mesh.vertices[mesh.faces] + np.asarray(shift)
        light = np.array((-.4, -.3, 1.0)); light /= np.linalg.norm(light)
        shade = .3 + .7 * np.maximum(mesh.face_normals @ light, 0)
        colors = np.asarray(to_rgb(color))[None, :] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors, edgecolors="none", alpha=1))
    ax.set(xlim=(-95, 85), ylim=(-50, 50), zlim=(0, 90), xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    ax.set_box_aspect((180, 100, 90))
    ax.view_init(elev=30, azim=-62)
    ax.set_title("Original printable D-peg and blind socket", pad=20)
    bx = fig.add_axes((.61, .34, .36, .49))
    theta = np.linspace(0, 2 * math.pi, 512)
    for r, flat, label, color in ((12, 6, "Peg D section", "#dc9e20"),
                                 (12.75, 6.75, "Bore: 0.75 mm clearance", "#405d79")):
        xy = np.array([radial_profile(t, r, flat) for t in theta])
        bx.plot(xy[:, 0], xy[:, 1], label=label, color=color, linewidth=2)
    bx.annotate("Flat outward normal +X", xy=(6, 0), xytext=(18, 0),
                arrowprops={"arrowstyle": "->"}, ha="left")
    bx.set(aspect="equal", xlim=(-18, 47), ylim=(-23, 22), xlabel="X (mm)", ylabel="Y (mm)")
    bx.grid(alpha=.2)
    bx.legend(loc="upper left")
    bx.set_title("Insertion cross section; all dimensions in mm")
    fig.text(.61, .10, "Shaft: R12, flat x=6, L35, tip chamfer 1\nGrip: diameter 50, height 50; total height 85\nSocket: 80 x 80 x 50; blind depth 30\nEntrance chamfer: 1; target insertion: 28", fontsize=11, linespacing=1.7)
    fig.savefig(PACKAGE / "preview.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    for folder in ("cad", "meshes"):
        (PACKAGE / folder).mkdir(parents=True, exist_ok=True)
    shaft = d_solid(SHAFT_RADIUS, SHAFT_FLAT, SHAFT_LENGTH).faces("<Z").edges().chamfer(TIP_CHAMFER)
    grip = cq.Workplane("XY").workplane(offset=SHAFT_LENGTH).circle(GRIP_RADIUS).extrude(GRIP_HEIGHT)
    peg = shaft.union(grip)
    radius, flat = SHAFT_RADIUS + CLEARANCE, SHAFT_FLAT + CLEARANCE
    low = d_solid(radius, flat, 1, SOCKET_HEIGHT - ENTRY_CHAMFER - 1).faces(">Z").wires().val()
    high = d_solid(radius + ENTRY_CHAMFER, flat + ENTRY_CHAMFER, 1, SOCKET_HEIGHT - 1).faces(">Z").wires().val()
    lead_in = cq.Solid.makeLoft([low, high], ruled=True)
    socket = cq.Workplane("XY").box(SOCKET_WIDTH, SOCKET_WIDTH, SOCKET_HEIGHT, centered=(True, True, False))
    socket = socket.cut(d_solid(radius, flat, HOLE_DEPTH, SOCKET_HEIGHT - HOLE_DEPTH)).cut(lead_in)
    geometries = {}
    for name, shape in (("peg", peg), ("socket", socket), ("shaft_collision", shaft), ("grip_collision", grip)):
        assert shape.val().isValid() and len(shape.solids().vals()) == 1
        geometries[name] = mesh_of(shape)
        geometries[name].export(PACKAGE / "meshes" / f"{name}.stl")
        if name in ("peg", "socket"):
            cq.exporters.export(shape, str(PACKAGE / "cad" / f"{name}.step"))
    bottom = trimesh.creation.box(extents=(SOCKET_WIDTH, SOCKET_WIDTH, SOCKET_HEIGHT - HOLE_DEPTH))
    bottom.apply_translation((0, 0, (SOCKET_HEIGHT - HOLE_DEPTH) / 2))
    walls = socket_wall_colliders()
    socket_colliders = [("Bottom", bottom), *walls]
    write_usd(PACKAGE / "peg.usd", "DPeg", geometries["peg"],
              [("Shaft", geometries["shaft_collision"]), ("Grip", geometries["grip_collision"])], kinematic=False, mass=.1)
    write_usd(PACKAGE / "socket.usd", "DSocket", geometries["socket"], socket_colliders, kinematic=True, mass=.5)
    np.savez_compressed(PACKAGE / "meshes" / "socket_collision_cells.npz", **{
        key: value for name, mesh in socket_colliders for key, value in
        ((name + "_vertices_mm", mesh.vertices), (name + "_faces", mesh.faces))
    })
    metadata = {
        "schema_version": 1, "provenance": "Original parametric geometry authored for this project; no external model source.",
        "units": {"usd": "m", "cad_step": "mm", "stl": "mm", "collision_npz": "mm"},
        "up_axis": "Z", "flat_normal_local": [1, 0, 0],
        "dimensions_m": {"shaft_radius": .012, "shaft_flat_x": .006, "shaft_length": .035,
                         "tip_chamfer": .001, "grip_diameter": .050, "grip_height": .050,
                         "socket_size": [.080, .080, .050], "hole_depth": .030,
                         "hole_radius": .01275, "hole_flat_x": .00675,
                         "clearance_per_side": .00075, "entry_chamfer": .001},
        "peg": {"usd": "peg.usd", "default_prim": "/DPeg", "mass_kg": .1,
                "tip_local": [0, 0, 0], "shaft_top_local": [0, 0, .035],
                "grip_center_local": [0, 0, .060], "top_local": [0, 0, .085],
                "visual_mesh_path": "VisualMesh", "tactile_mesh_path": "TactileSurface",
                "collision_mesh_paths": ["Collisions/Shaft", "Collisions/Grip"],
                "kinematic": False},
        "socket": {"usd": "socket.usd", "default_prim": "/DSocket", "kinematic": True,
                   "mouth_local": [0, 0, .050], "hole_bottom_local": [0, 0, .020],
                   "target_depth_m": .028, "target_tip_local": [0, 0, .022],
                   "visual_mesh_path": "VisualMesh", "tactile_mesh_path": "TactileSurface",
                   "collision_cell_count": len(socket_colliders),
                   "collision_strategy": "Explicit Bottom plus angular convex Wall cells, including entrance chamfer; never a hull of the whole socket."},
        "physics": {"contact_offset_m": .0001, "rest_offset_m": 0.0,
                    "friction": {"grip": {"static": .9, "dynamic": .8},
                                 "shaft_socket": {"static": .5, "dynamic": .4}},
                    "peg_joint_count": 0, "collision_approximation": "convexHull per explicit convex piece"},
        "geometry": {name: fingerprint(mesh) for name, mesh in geometries.items()},
        "tessellation_tolerance_m": TESSELLATION_TOLERANCE_MM * .001,
        "notes": ["STL is unitless: import with millimetre units.",
                  "0.75 mm clearance is geometric clearance at the circular arc and flat; printing tolerances require physical calibration.",
                  "The paint-like +X arrow is visual only; it is excluded from the canonical tactile mesh and collision.",
                  "At target depth 28 mm the peg tip remains 2 mm above the blind bottom and the grip shoulder is 7 mm above the mouth."]
    }
    for name in ("peg", "socket"):
        metadata[name]["usd_sha256"] = hashlib.sha256((PACKAGE / f"{name}.usd").read_bytes()).hexdigest()
    (PACKAGE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    preview(geometries["peg"], geometries["socket"])
    print(json.dumps({"output": str(PACKAGE), "socket_collision_cells": len(socket_colliders),
                      "peg_bounds_mm": geometries["peg"].bounds.tolist(),
                      "socket_bounds_mm": geometries["socket"].bounds.tolist()}))


if __name__ == "__main__":
    main()
