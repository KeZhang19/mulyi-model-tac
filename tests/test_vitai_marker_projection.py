from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.project_vitai_markers_to_mesh import (  # noqa: E402
    FINGERS,
    RAY_COLS,
    RAY_ROWS,
    camera_xy_to_ray_grid_coordinates,
    closest_mesh_point_to_ray,
    compose_camera_plane_ray_grid,
    compose_exact_marker_camera_plane_rays,
    element_transform,
    marker_surface_hit_indices,
    order_closed_boundary_segments_xy,
    outer_surface_hit_indices,
    project_finger,
    sample_camera_visible_surface,
)


LAYOUT_PATH = (
    REPO_ROOT
    / "assets"
    / "revo21_right_touch"
    / "marker_positions"
    / "vitai_4fingers"
    / "marker_positions.npz"
)


def test_closest_mesh_point_to_ray_returns_triangle_boundary():
    triangles = np.array(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]],
        dtype=np.float64,
    )

    triangle_id, point, error = closest_mesh_point_to_ray(
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        triangles,
    )

    assert triangle_id == 0
    np.testing.assert_allclose(point, [1.0, 0.0, 1.0])
    assert error == 1.0


def test_vitai_marker_projection_uses_outer_rubber_surface():
    triangles = np.array(
        [
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
            [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]],
        ],
        dtype=np.float64,
    )
    projected = project_finger(
        marker_ids=np.array(["M002"]),
        pixels=np.array([[0.0, 0.0]]),
        rays_camera=np.array([[0.0, 0.0, 1.0]]),
        camera_matrix=np.eye(3),
        distortion=np.zeros(5),
        distortion_valid=np.array([True]),
        monotonic_radial_scale=1.0,
        camera_origin_surface=np.zeros(3),
        camera_rotation_surface=np.eye(3),
        triangles=triangles,
        triangle_normals=np.array([[0.0, 0.0, 1.0]] * 2),
    )

    np.testing.assert_allclose(projected["points_camera_m"][0], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(projected["normals_link"][0], [0.0, 0.0, 1.0])


def test_camera_visible_surface_sampling_keeps_uniform_outer_hits_in_image():
    triangles = np.array(
        [
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0]],
            [[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
            [[-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [2.0, 2.0, 2.0]],
            [[-2.0, -2.0, 2.0], [2.0, 2.0, 2.0], [-2.0, 2.0, 2.0]],
        ],
        dtype=np.float64,
    )
    points, normals, visible = sample_camera_visible_surface(
        triangles,
        np.tile(np.array([[0.0, 0.0, 1.0]]), (4, 1)),
        camera_origin_surface=np.zeros(3),
        camera_rotation_surface=np.eye(3),
        camera_matrix=np.array(
            [[10.0, 0.0, 50.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]]
        ),
        distortion=np.zeros(5),
        image_size=(100, 100),
        sample_count=2000,
        seed=7,
    )

    assert len(points) > 0
    np.testing.assert_allclose(points[:, 2], 2.0)
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (len(normals), 1)))
    assert np.all(visible)


def test_vitai_marker_layout_contains_all_fingers_with_quality_flags():
    with np.load(LAYOUT_PATH, allow_pickle=False) as layout:
        marker_ids = layout["marker_ids"].astype(str)
        distortion_valid = layout["distortion_valid"].astype(bool)
        distortion_model = layout["distortion_model"].astype(str)
        selected_reprojection = layout["selected_model_reprojection_error_px"]

        assert marker_ids.shape == (100,)
        assert marker_ids[0] == "M002"
        assert marker_ids[-1] == "M101"
        assert "M001" not in marker_ids
        assert np.count_nonzero(distortion_valid) == 87
        assert set(distortion_model) == {
            "brown_exact",
            "monotonic_extrapolation",
        }
        assert np.max(selected_reprojection[distortion_valid]) <= 0.5
        assert np.max(selected_reprojection[~distortion_valid]) < 1.0e-6
        extrapolated_pixels = layout["pixels_undistorted"][~distortion_valid]
        assert len(np.unique(np.round(extrapolated_pixels, 5), axis=0)) == 13

        for finger in ("middle", "index", "ring", "pinky", "thumb"):
            points = layout[f"{finger}_points_link_m"]
            normals = layout[f"{finger}_normals_link"]
            distance = layout[f"{finger}_distance_m"]
            methods = layout[f"{finger}_method"].astype(str)
            reprojection = layout[f"{finger}_mesh_reprojection_error_px"]

            assert points.shape == (100, 3)
            assert normals.shape == (100, 3)
            assert np.isfinite(points).all()
            assert np.isfinite(normals).all()
            max_distance = 0.021 if finger == "thumb" else 0.017
            assert np.all((distance > 0.008) & (distance < max_distance))
            assert np.median(distance) > 0.011
            assert np.count_nonzero(methods == "ray_hit") >= 98
            ray_hits = methods == "ray_hit"
            assert np.max(reprojection[ray_hits]) <= 0.5


def test_base_camera_plane_layout_stays_regular_and_marker_rays_are_separate():
    hull = np.array(
        [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
        dtype=np.float32,
    )
    starts, directions = compose_camera_plane_ray_grid(
        hull,
        camera_origin_link=np.array([1.0, 2.0, 3.0]),
        camera_rotation_link=np.eye(3),
        rows=32,
        cols=24,
    )

    assert starts.shape == (32, 24, 3)
    assert directions.shape == starts.shape
    np.testing.assert_allclose(starts[..., 2], 3.0)
    np.testing.assert_allclose(directions, np.broadcast_to([0.0, 0.0, 1.0], starts.shape))
    assert np.all(np.diff(starts[..., 0], axis=1) > 0.0)
    assert np.all(np.diff(starts[..., 1], axis=0) > 0.0)

    marker_starts, marker_directions, marker_valid = compose_exact_marker_camera_plane_rays(
        np.array([[0.25, -0.5, 2.0], [np.nan, np.nan, np.nan]], dtype=np.float32),
        np.array([True, False]),
        camera_origin_link=np.array([1.0, 2.0, 3.0]),
        camera_rotation_link=np.eye(3),
    )
    np.testing.assert_allclose(marker_starts[0], [1.25, 1.5, 3.0])
    np.testing.assert_allclose(marker_starts[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(marker_directions, [[0.0, 0.0, 1.0]] * 2)
    np.testing.assert_array_equal(marker_valid, [True, False])


def test_marker_surface_hit_selection_can_mix_first_and_second_hits():
    triangles = np.array(
        [
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
            [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]],
        ],
        dtype=np.float64,
    )
    starts = np.zeros((2, 3), dtype=np.float32)
    directions = np.broadcast_to([0.0, 0.0, 1.0], starts.shape).copy()
    hit_indices = marker_surface_hit_indices(
        starts,
        directions,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32),
        np.array([True, True]),
        triangles,
    )

    np.testing.assert_array_equal(hit_indices, [1, 2])


def test_structured_outer_surface_hit_selection_uses_farthest_available_hit():
    triangles = np.array(
        [
            [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
            [[-0.25, -0.25, 2.0], [0.25, -0.25, 2.0], [0.0, 0.25, 2.0]],
        ],
        dtype=np.float64,
    )
    starts = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.0]], dtype=np.float32)
    directions = np.broadcast_to([0.0, 0.0, 1.0], starts.shape).copy()

    hit_indices = outer_surface_hit_indices(starts, directions, triangles)

    np.testing.assert_array_equal(hit_indices, [2, 1])


def test_unordered_boundary_segments_form_a_canonical_polygon():
    segments = np.array(
        [
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )

    polygon = order_closed_boundary_segments_xy(segments)

    np.testing.assert_allclose(
        polygon,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )


def test_rl_ours_uses_separate_base_and_marker_camera_plane_ray_layouts():
    config_path = (
        REPO_ROOT
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    source = config_path.read_text(encoding="utf-8")

    assert "ray_layout_npz=" in source
    assert "ray_layout_prefix=" in source
    assert 'finger if tactile_implementation == "ours" else None' in source
    assert 'marker_prefix = f"{finger}_marker"' in source
    assert "TIANJI_HYDROSHEAR_MARKER_SURFACE_SENSOR_NAMES" in source
    assert "TIANJI_HYDROSHEAR_MARKER_OBJECT_SENSOR_NAMES" in source
    assert "TIANJI_TACMAP_CAMERA_PLANE_MAX_DISTANCE_M" in source
    assert "use_ray_hit_index_layout=" in source


def test_generated_camera_plane_base_is_regular_and_marker_rays_are_exact():
    def inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        starts = polygon
        ends = np.roll(polygon, -1, axis=0)
        point_x = points[:, 0, None]
        point_y = points[:, 1, None]
        crosses_y = (starts[None, :, 1] > point_y) != (ends[None, :, 1] > point_y)
        edge_x = starts[None, :, 0] + (
            (point_y - starts[None, :, 1])
            * (ends[None, :, 0] - starts[None, :, 0])
            / (ends[None, :, 1] - starts[None, :, 1] + 1.0e-30)
        )
        return np.logical_xor.reduce(crosses_y & (point_x < edge_x), axis=1)

    with np.load(LAYOUT_PATH, allow_pickle=False) as layout:
        pixels = np.asarray(layout["pixels_distorted"], dtype=np.float32)
        distortion_valid = np.asarray(layout["distortion_valid"], dtype=bool)
        for finger in FINGERS:
            starts_link = np.asarray(
                layout[f"{finger}_ray_starts_link_m"], dtype=np.float32
            )
            directions_link = np.asarray(
                layout[f"{finger}_ray_directions_link"], dtype=np.float32
            )
            marker_mask = np.asarray(layout[f"{finger}_ray_is_marker"], dtype=bool)
            surface_hit_index = np.asarray(
                layout[f"{finger}_ray_surface_hit_index"], dtype=np.uint8
            )
            marker_starts_link = np.asarray(
                layout[f"{finger}_marker_ray_starts_link_m"], dtype=np.float32
            )
            marker_directions_link = np.asarray(
                layout[f"{finger}_marker_ray_directions_link"], dtype=np.float32
            )
            marker_ray_valid = np.asarray(layout[f"{finger}_marker_ray_valid"], dtype=bool)
            marker_surface_hit_index = np.asarray(
                layout[f"{finger}_marker_ray_surface_hit_index"], dtype=np.uint8
            )
            hull = np.asarray(
                layout[f"{finger}_camera_visible_hull_xy_m"], dtype=np.float32
            )
            camera_range_boundary = np.asarray(
                layout[f"{finger}_camera_range_aug4_boundary_xy_m"], dtype=np.float32
            )
            camera_origin = np.asarray(
                layout[f"{finger}_camera_origin_link_m"], dtype=np.float32
            )
            camera_rotation = np.asarray(
                layout[f"{finger}_camera_rotation_link"], dtype=np.float32
            )
            starts_camera = (starts_link - camera_origin) @ camera_rotation
            directions_camera = directions_link @ camera_rotation
            marker_starts_camera = (marker_starts_link - camera_origin) @ camera_rotation
            marker_directions_camera = marker_directions_link @ camera_rotation

            assert starts_link.shape == (RAY_ROWS, RAY_COLS, 3)
            assert directions_link.shape == starts_link.shape
            assert surface_hit_index.shape == (RAY_ROWS, RAY_COLS)
            assert np.all((surface_hit_index == 1) | (surface_hit_index == 2))
            assert np.count_nonzero(marker_mask) == 0
            assert np.count_nonzero(surface_hit_index == 1) > 0
            assert np.count_nonzero(surface_hit_index == 2) > 0
            np.testing.assert_array_equal(hull, camera_range_boundary)
            np.testing.assert_allclose(starts_camera[..., 2], 0.0, atol=1.0e-7)
            np.testing.assert_allclose(
                directions_camera,
                np.broadcast_to([0.0, 0.0, 1.0], directions_camera.shape),
                atol=1.0e-7,
            )
            assert np.all(
                inside_polygon(
                    starts_camera[..., :2].reshape(-1, 2),
                    camera_range_boundary,
                )
            )
            assert np.all(np.diff(starts_camera[..., 0], axis=1) > 0.0)
            assert np.all(np.diff(starts_camera[..., 1], axis=0) > 0.0)

            methods = np.asarray(layout[f"{finger}_method"]).astype(str)
            marker_points_camera = np.asarray(
                layout[f"{finger}_points_camera_m"], dtype=np.float32
            )
            exact = distortion_valid & (methods == "ray_hit")
            exact_ids = np.flatnonzero(exact)
            assert marker_starts_link.shape == (100, 3)
            assert marker_directions_link.shape == marker_starts_link.shape
            assert marker_surface_hit_index.shape == (100,)
            np.testing.assert_array_equal(marker_ray_valid, exact)
            assert np.all((marker_surface_hit_index == 1) | (marker_surface_hit_index == 2))
            np.testing.assert_allclose(marker_starts_camera[:, 2], 0.0, atol=1.0e-7)
            np.testing.assert_allclose(
                marker_directions_camera,
                np.broadcast_to([0.0, 0.0, 1.0], marker_directions_camera.shape),
                atol=1.0e-7,
            )
            np.testing.assert_allclose(
                marker_starts_camera[exact_ids, :2],
                marker_points_camera[exact_ids, :2],
                atol=1.0e-7,
            )


def test_marker_mapping_reports_camera_range_membership_and_coordinates():
    with np.load(LAYOUT_PATH, allow_pickle=False) as layout:
        for finger in FINGERS:
            coordinates, inside = camera_xy_to_ray_grid_coordinates(
                layout[f"{finger}_camera_visible_hull_xy_m"],
                layout[f"{finger}_points_camera_m"][:, :2],
                rows=RAY_ROWS,
                cols=RAY_COLS,
            )
            marker_valid = np.asarray(layout[f"{finger}_marker_ray_valid"], dtype=bool)

            assert np.count_nonzero(inside & marker_valid) > 0
            assert np.all(np.isfinite(coordinates[inside]))
            assert np.all(~np.isfinite(coordinates[~inside]))
            assert np.all(coordinates[inside, 0] >= -0.5 - 1.0e-6)
            assert np.all(coordinates[inside, 0] <= RAY_ROWS - 0.5 + 1.0e-6)
            assert np.all(coordinates[inside, 1] >= -0.5 - 1.0e-6)
            assert np.all(coordinates[inside, 1] <= RAY_COLS - 0.5 + 1.0e-6)


def test_marker_camera_coordinates_round_trip_to_link():
    with np.load(LAYOUT_PATH, allow_pickle=False) as layout:
        for finger in FINGERS:
            points_link = layout[f"{finger}_points_link_m"]
            points_camera = layout[f"{finger}_points_camera_m"]
            camera_origin = layout[f"{finger}_camera_origin_link_m"]
            camera_rotation = layout[f"{finger}_camera_rotation_link"]

            reconstructed = points_camera @ camera_rotation.T + camera_origin
            np.testing.assert_allclose(reconstructed, points_link, atol=1.0e-7)


def test_vitai_marker_normals_point_out_of_surface_away_from_camera():
    urdf = (
        REPO_ROOT
        / "assets"
        / "revo21_right_touch"
        / "urdf"
        / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )
    joints = {
        joint.attrib["name"]: joint
        for joint in ET.parse(urdf).getroot().findall("joint")
    }

    with np.load(LAYOUT_PATH, allow_pickle=False) as layout:
        for finger, config in FINGERS.items():
            camera_xyz, _ = element_transform(joints[config["camera_joint"]])
            surface_xyz, surface_rotation = element_transform(
                joints[config["surface_joint"]]
            )
            camera_origin_surface = surface_rotation.T @ (camera_xyz - surface_xyz)
            points = layout[f"{finger}_points_link_m"]
            normals = layout[f"{finger}_normals_link"]
            camera_facing = np.einsum(
                "ij,ij->i",
                normals,
                camera_origin_surface - points,
            )

            assert np.all(camera_facing < 0.0), (
                finger,
                int(np.count_nonzero(camera_facing >= 0.0)),
            )
