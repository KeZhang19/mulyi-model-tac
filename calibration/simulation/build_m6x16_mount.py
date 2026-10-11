#!/usr/bin/env python3
"""Create a V4-derived mount with deeper counterbores; never overwrite inputs.

CPU only: python3 calibration/simulation/build_m6x16_mount.py --output NEW_DIR
The default scene is updated separately, after inspecting validation.json.
"""
import argparse
import json
import math
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from build_m6x20_mount import (
    V4, MOUNT_HOLES_MM, connected_components, repair_occ_tessellation,
    sha256, write_3mf,
)


def build_m6x16_mount(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    source_path = V4 / 'cad/lower_calibration_v4.brep'
    source = cq.importers.importBrep(str(source_path)).val()
    mount = source
    # Existing counterbores start at Z=24. Keep a 10 mm bearing floor.
    for x, y in MOUNT_HOLES_MM:
        mount = mount.cut(cq.Solid.makeCylinder(6, 50, cq.Vector(x, y, 10)))
    mount = mount.clean()
    assert mount.isValid() and len(mount.Solids()) == 1
    stem = 'lower_calibration_v4_m6x16'
    mount.exportBrep(str(output / f'{stem}.brep'))
    cq.exporters.export(mount, str(output / f'{stem}.step'))
    stl = output / f'{stem}_print.stl'
    cq.exporters.export(mount, str(stl), tolerance=0.02, angularTolerance=0.05)
    # Cutting the unshortened V4 also exposes four triangular tessellation gaps.
    # Collapse only sub-0.002 mm edges in 3-edge loops; the existing
    # recovery routine then handles the two inherited quadrilateral gaps.
    raw = trimesh.load(stl, force='mesh', process=True)
    raw.update_faces(raw.area_faces > 1e-12)
    raw.update_faces(raw.unique_faces())
    counts = np.bincount(raw.edges_unique_inverse)
    neighbors = {}
    for a, b in raw.edges_unique[counts == 1]:
        neighbors.setdefault(int(a), set()).add(int(b))
        neighbors.setdefault(int(b), set()).add(int(a))
    triangles = set()
    for a, adjacent in neighbors.items():
        if len(adjacent) == 2:
            b, c = sorted(adjacent)
            if c in neighbors[b]:
                triangles.add(tuple(sorted((a, b, c))))
    assert len(triangles) == 4
    gap_error = 0.0
    for a, b, c in sorted(triangles):
        pairs = [(a, b), (b, c), (c, a)]
        left, right = min(pairs, key=lambda pair: np.linalg.norm(raw.vertices[pair[0]]-raw.vertices[pair[1]]))
        gap_error = max(gap_error, float(np.linalg.norm(raw.vertices[left]-raw.vertices[right])))
        raw.vertices[right] = raw.vertices[left]
    assert gap_error < .002
    raw.merge_vertices()
    raw.update_faces(raw.area_faces > 1e-12)
    raw.export(stl)
    mesh, recovery = repair_occ_tessellation(mount, stl)
    recovery.append(dict(triangular_gaps=4, max_collapsed_edge_length_mm=gap_error,
                         acceptance_tolerance_mm=.002))
    write_3mf(mesh, output / f'{stem}_print.3mf')
    step = cq.importers.importStep(str(output / f'{stem}.step')).val()
    assert step.isValid() and len(step.Solids()) == 1
    assert abs(step.Volume() - mount.Volume()) / mount.Volume() < 1e-5
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert connected_components(mesh) == 1
    assert abs(mesh.volume - mount.Volume()) / mount.Volume() < 1e-3
    expected_removed = 4 * math.pi * (6**2 - 3.4**2) * 14
    removed_solid_volume = source.cut(mount).Volume()
    assert abs(removed_solid_volume - expected_removed) < 1e-3
    # All geometry outside the four annular cuts is identical by construction;
    # verify no additions and the unchanged original bounds.
    assert mount.cut(source).Volume() < 1e-6
    # Compare like-for-like analytic BREP bounds without cached STL triangulation.
    source_bounds = cq.importers.importBrep(str(source_path)).val().BoundingBox()
    readback_bounds = cq.importers.importBrep(str(output / f'{stem}.brep')).val().BoundingBox()
    for attr in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'):
        assert abs(getattr(source_bounds, attr) - getattr(readback_bounds, attr)) < 1e-6
    holes = []
    for x, y in MOUNT_HOLES_MM:
        probes = 0
        for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            for r, z, inside in ((3.35, .1, False), (3.35, 9.9, False),
                                  (3.45, 9.9, True), (5.95, 9.9, True),
                                  (5.95, 10.1, False), (6.05, 10.1, True),
                                  (5.95, 23.9, False), (6.05, 23.9, True)):
                assert mount.isInside(cq.Vector(x+r*math.cos(theta), y+r*math.sin(theta), z), 1e-7) == inside
                probes += 1
        # Assumed cylindrical head diameter 10 / height 6; length under head 16.
        screw = cq.Solid.makeCylinder(3, 16, cq.Vector(x, y, -6)).fuse(
            cq.Solid.makeCylinder(5, 6, cq.Vector(x, y, 10)))
        interference = mount.intersect(screw).Volume()
        assert interference < 1e-6
        # Verify straight top insertion access up to above the entire part.
        access = mount.intersect(cq.Solid.makeCylinder(5, 50, cq.Vector(x, y, 10))).Volume()
        assert access < 1e-6
        # STL ray checks at annular bearing position recover both bottom and floor.
        # Vertical ray / triangle barycentrics, without optional rtree dependency.
        tri = mesh.triangles
        edge1, edge2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
        delta = np.array([x+4.5, y])-tri[:, 0, :2]
        det = edge1[:, 0]*edge2[:, 1]-edge1[:, 1]*edge2[:, 0]
        valid = abs(det) > 1e-12
        u = np.full(len(det), -1.0)
        v = u.copy()
        u[valid] = (delta[valid, 0]*edge2[valid, 1]-delta[valid, 1]*edge2[valid, 0])/det[valid]
        v[valid] = (edge1[valid, 0]*delta[valid, 1]-edge1[valid, 1]*delta[valid, 0])/det[valid]
        hit = valid & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
        hit_z = np.unique(np.round((tri[:, 0, 2]+u*edge1[:, 2]+v*edge2[:, 2])[hit], 6))
        assert np.allclose(hit_z, [0, 10], atol=.02)
        holes.append(dict(xy_mm=[x, y], cad_probes_passed=probes,
                          screw_intersection_mm3=interference, insertion_intersection_mm3=access,
                          mesh_bearing_ray_z_mm=hit_z.tolist()))
    report = dict(schema='calibration_mount_m6x16_v1', source_brep=str(source_path),
                  source_sha256=sha256(source_path), counterbore_diameter_mm=12,
                  through_diameter_mm=6.8, original_bearing_z_mm=24,
                  bearing_z_mm=10, additional_cut_depth_mm=14,
                  screw_length_under_head_mm=16, assumed_head_diameter_mm=10,
                  assumed_head_height_mm=6, projection_no_washer_mm=6,
                  projection_with_1mm_washer_mm=5,
                  nominal_projection_is_not_verified_thread_engagement=True,
                  physical_validation=False, load_capacity_verified=False,
                  holes=holes, cad_valid=True, cad_solids=1,
                  cad_volume_mm3=mount.Volume(), removed_volume_mm3=source.Volume()-mount.Volume(),
                  step_volume_relative_error=abs(step.Volume()-mount.Volume())/mount.Volume(),
                  step_volume_relative_tolerance=1e-5,
                  expected_removed_volume_mm3=expected_removed,
                  removed_boolean_volume_mm3=removed_solid_volume,
                  stl_watertight=True, stl_winding_consistent=True, stl_components=1,
                  stl_cad_volume_relative_error=abs(mesh.volume-mount.Volume())/mount.Volume(),
                  bounds_mm=mesh.bounds.tolist(), recovered_source_patches=recovery,
                  outputs_sha256={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    (output / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build_m6x16_mount(parser.parse_args().output.resolve())
