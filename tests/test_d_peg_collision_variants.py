"""Geometry acceptance checks for optional D-peg socket collision candidates."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "assets/d_peg_insertion"
SPEC = importlib.util.spec_from_file_location("d_peg_collision_builder", PACKAGE / "tools/optimize_collisions.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def collision_hulls(text):
    import re
    body = text.split(BUILDER.COLLISION_SCOPE, 1)[1]
    names = re.findall(r'def Mesh "([^"]+)"', body)
    return [ConvexHull(BUILDER.mesh_arrays(BUILDER.mesh_block(text, name))[0]).equations for name in names]


def inside_any_collider(points, hulls):
    inside = np.zeros(len(points), dtype=bool)
    for equations in hulls:
        inside |= np.all(points @ equations[:, :3].T + equations[:, 3] <= 1e-8, axis=1)
    return inside


@pytest.mark.parametrize("segments,colliders", [(48, 36), (32, 26)])
def test_candidates_preserve_noncollision_data_and_floor(segments, colliders):
    source = (PACKAGE / "socket.usd").read_text()
    candidate = (PACKAGE / f"collision_variants/socket_{segments}.usd").read_text()
    assert candidate.split(BUILDER.COLLISION_SCOPE, 1)[0] == source.split(BUILDER.COLLISION_SCOPE, 1)[0]
    assert BUILDER.mesh_block(candidate, "Bottom") == BUILDER.mesh_block(source, "Bottom")
    assert candidate.count("bool physics:collisionEnabled = true") == colliders
    assert candidate.count('physics:approximation = "convexHull"') == colliders


@pytest.mark.parametrize("segments", [48, 32])
def test_full_size_aligned_shaft_fits_but_offset_and_wrong_yaw_are_blocked(segments):
    text = (PACKAGE / f"collision_variants/socket_{segments}.usd").read_text()
    hulls = collision_hulls(text)
    theta = np.linspace(0, 2 * np.pi, 2048, endpoint=False)
    # Independent section construction: parametric D boundary, with the full
    # shaft radius throughout (more conservative than the real chamfered tip).
    xy = np.stack((12 * np.cos(theta), 12 * np.sin(theta)), axis=1)
    xy[:, 0] = np.minimum(xy[:, 0], 6.0)
    shaft = np.concatenate([np.column_stack((xy, np.full(len(xy), z))) for z in np.linspace(22, 49.9, 12)])
    assert not inside_any_collider(shaft, hulls).any()
    offset = shaft.copy()
    offset[:, 0] += 5.0
    assert inside_any_collider(offset, hulls).any()
    wrong_yaw = shaft.copy()
    wrong_yaw[:, :2] *= -1
    assert inside_any_collider(wrong_yaw, hulls).any()
    assert inside_any_collider(np.array([[0, 0, 19.9]]), hulls).all()
    assert not inside_any_collider(np.array([[0, 0, 20.1], [0, 0, 50.1]]), hulls).any()


@pytest.mark.parametrize("segments,max_error_um", [(48, 29.45), (32, 66.22)])
def test_circle_and_flat_chamfer_have_bounded_error(segments, max_error_um):
    dimensions = json.loads((PACKAGE / "metadata.json").read_text())["dimensions_m"]
    angles, cells = BUILDER.wall_cells(segments, dimensions)
    report = BUILDER.cavity_validation(angles, cells, dimensions, 4096)
    assert report["max_inward_error_um"] < max_error_um
    assert report["max_outward_error_um"] < 1e-4
    assert report["flat_x_max_error_um"] < 1e-4


def test_triangle_candidate_matches_visual_mesh_without_closing_hole():
    text = (PACKAGE / "collision_variants/socket_triangle.usd").read_text()
    visual_vertices, visual_faces = BUILDER.mesh_arrays(BUILDER.mesh_block(text, "VisualMesh"))
    collision_vertices, collision_faces = BUILDER.mesh_arrays(BUILDER.mesh_block(text, "SocketMesh"))
    np.testing.assert_allclose(collision_vertices, visual_vertices, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(collision_faces, visual_faces)
    assert text.count('physics:approximation = "none"') == 1


def test_contact_acceptance_captures_impulses_missed_by_trajectory_stride():
    spec = importlib.util.spec_from_file_location("insertion_physics_validator", PACKAGE / "tools/validate_insertion_physics.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    # Matches the observed failure mode: the 20 Hz trajectory samples always
    # land on zero, despite contact during every other 240 Hz physics step.
    forces = [0.0 if step % 2 == 0 else .445937 for step in range(1680)]
    report = validator.contact_force_peaks(forces, forces[::12])
    assert report["maximum_socket_contact_n"] > .05
    assert report["trajectory_sampled_maximum_socket_contact_n"] == 0.0
    assert report["full_rate_contact_sample_count"] == 1680
    assert report["trajectory_contact_sample_count"] == 140
    absent = validator.contact_force_peaks([0.0] * 1680, [0.0] * 140)
    assert absent["maximum_socket_contact_n"] == 0.0
