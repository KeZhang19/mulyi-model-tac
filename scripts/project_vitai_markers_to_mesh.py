#!/usr/bin/env python3
"""Project calibrated Vitai marker pixels onto the fingertip rubber meshes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tacmap.scripts.generate_tactile_map_from_usd import (  # noqa: E402
    load_stl_triangles,
    normals_from_triangles,
    ray_triangle_nearest,
)


FINGERS = {
    "middle": {
        "camera_joint": "right_middip_camera_joint",
        "surface_joint": "right_middip_roll_rubber_joint",
        "surface_link": "right_middip_roll_rubber_link",
    },
    "index": {
        "camera_joint": "right_index_camera",
        "surface_joint": "right_indexdip_roll_rubber_joint",
        "surface_link": "right_indexdip_roll_rubber_link",
    },
    "ring": {
        "camera_joint": "right_ring_camera_joint",
        "surface_joint": "right_ringdip_roll_rubber_joint",
        "surface_link": "right_ringdip_roll_rubber_link",
    },
    "pinky": {
        "camera_joint": "right_pinky_camera_joint",
        "surface_joint": "right_pinkydip_roll_rubber_joint",
        "surface_link": "right_pinkydip_roll_rubber_link",
    },
    "thumb": {
        "camera_joint": "right_thumbdip_camera_joint",
        "surface_joint": "right_thumbdip_roll_rubber_joint",
        "surface_link": "right_thumbdip_roll_rubber_link",
    },
}
RAY_ROWS = 32
RAY_COLS = 24


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def element_transform(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    origin = None if element is None else element.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    return xyz, rpy_matrix(rpy)


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
    image_size = tuple(int(value) for value in data["image_size"])
    if matrix.shape != (3, 3) or distortion.shape != (5,) or len(image_size) != 2:
        raise ValueError(f"Unsupported camera calibration in {path}")
    return matrix, distortion, image_size


def load_marker_pixels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    marker_ids = np.asarray([row["marker_id"] for row in rows])
    pixels = np.asarray(
        [[float(row["u_px"]), float(row["v_px"])] for row in rows],
        dtype=np.float64,
    )
    if len(marker_ids) == 0 or len(set(marker_ids.tolist())) != len(marker_ids):
        raise ValueError(f"Marker IDs must be non-empty and unique in {path}")
    return marker_ids, pixels


def distort_normalized(
    points: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    k1, k2, p1, p2, k3 = distortion
    radius_sq = x * x + y * y
    radial = (
        1.0
        + k1 * radius_sq
        + k2 * radius_sq * radius_sq
        + k3 * radius_sq * radius_sq * radius_sq
    )
    return np.column_stack(
        (
            x * radial
            + 2.0 * p1 * x * y
            + p2 * (radius_sq + 2.0 * x * x),
            y * radial
            + p1 * (radius_sq + 2.0 * y * y)
            + 2.0 * p2 * x * y,
        )
    )


def fit_monotonic_radial_scale(
    distortion: np.ndarray,
    max_distorted_radius: float,
) -> tuple[float, float, float]:
    """Fit an invertible atan radial model to the calibrated Brown branch."""

    k1, k2, _p1, _p2, k3 = distortion
    roots = np.roots((7.0 * k3, 5.0 * k2, 3.0 * k1, 1.0))
    turning_radii = [
        np.sqrt(root.real)
        for root in roots
        if abs(root.imag) < 1.0e-10 and root.real > 0.0
    ]
    fit_radius = 0.85 * min(turning_radii) if turning_radii else 1.5
    radii = np.linspace(0.0, fit_radius, 1000)
    brown = radii * (
        1.0 + k1 * radii**2 + k2 * radii**4 + k3 * radii**6
    )
    upper = min(1.5, 0.98 * np.pi / (2.0 * max(max_distorted_radius, 1.0e-6)))
    result = least_squares(
        lambda value: np.arctan(value[0] * radii) / value[0] - brown,
        np.array([min(0.9, 0.9 * upper)]),
        bounds=(0.01, upper),
    )
    scale = float(result.x[0])
    rms = float(
        np.sqrt(np.mean((np.arctan(scale * radii) / scale - brown) ** 2))
    )
    return scale, float(fit_radius), rms


def distort_normalized_monotonic(
    points: np.ndarray,
    distortion: np.ndarray,
    radial_scale: float,
) -> np.ndarray:
    """Apply a monotonic continuation of the calibrated radial distortion."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    _k1, _k2, p1, p2, _k3 = distortion
    radius_sq = x * x + y * y
    radius = np.sqrt(radius_sq)
    radial = np.ones_like(radius)
    nonzero = radius > 1.0e-12
    radial[nonzero] = np.arctan(radial_scale * radius[nonzero]) / (
        radial_scale * radius[nonzero]
    )
    return np.column_stack(
        (
            x * radial
            + 2.0 * p1 * x * y
            + p2 * (radius_sq + 2.0 * x * x),
            y * radial
            + p1 * (radius_sq + 2.0 * y * y)
            + 2.0 * p2 * x * y,
        )
    )


def invert_distortion(
    pixels: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    max_reprojection_error_px: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Invert Brown exactly where possible and monotonically extrapolate edges."""

    target = np.column_stack(
        (
            (pixels[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (pixels[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
        )
    )
    normalized = np.empty_like(target)
    residual_px = np.empty(len(target), dtype=np.float64)
    focal = np.array((camera_matrix[0, 0], camera_matrix[1, 1]))

    for index, target_point in enumerate(target):
        target_radius = float(np.linalg.norm(target_point))
        unit = target_point / max(target_radius, 1.0e-12)
        candidates = []
        for radius in (0.0, 0.5, 1.0, 1.5, 1.85, 2.2):
            result = least_squares(
                lambda point: distort_normalized(point, distortion)[0]
                - target_point,
                unit * min(radius, max(target_radius * 1.5, radius)),
                bounds=(-2.5, 2.5),
                max_nfev=500,
                ftol=1.0e-13,
                xtol=1.0e-13,
                gtol=1.0e-13,
            )
            error = float(
                np.linalg.norm(
                    (distort_normalized(result.x, distortion)[0] - target_point)
                    * focal
                )
            )
            candidates.append((error, float(np.linalg.norm(result.x)), result.x))

        exact = [candidate for candidate in candidates if candidate[0] <= 1.0e-6]
        best = min(exact, key=lambda item: item[1]) if exact else min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        residual_px[index] = best[0]
        normalized[index] = best[2]

    valid = residual_px <= float(max_reprojection_error_px)
    radial_scale, fit_radius, fit_rms = fit_monotonic_radial_scale(
        distortion,
        float(np.max(np.linalg.norm(target, axis=1))),
    )
    selected_residual_px = residual_px.copy()
    for index in np.flatnonzero(~valid):
        target_point = target[index]
        target_radius = float(np.linalg.norm(target_point))
        unit = target_point / max(target_radius, 1.0e-12)
        initial_radius = np.tan(
            min(radial_scale * target_radius, np.pi / 2.0 - 0.02)
        ) / radial_scale
        result = least_squares(
            lambda point: distort_normalized_monotonic(
                point,
                distortion,
                radial_scale,
            )[0]
            - target_point,
            unit * initial_radius,
            bounds=(-20.0, 20.0),
            max_nfev=500,
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
        )
        normalized[index] = result.x
        selected_residual_px[index] = np.linalg.norm(
            (
                distort_normalized_monotonic(
                    result.x,
                    distortion,
                    radial_scale,
                )[0]
                - target_point
            )
            * focal
        )

    return (
        normalized,
        residual_px,
        selected_residual_px,
        valid,
        radial_scale,
        fit_radius,
        fit_rms,
    )


def load_surface_triangles(
    urdf_path: Path,
    link: ET.Element,
) -> tuple[Path, np.ndarray, np.ndarray]:
    visual = link.find("visual")
    mesh = None if visual is None else visual.find("./geometry/mesh")
    if mesh is None:
        raise ValueError(f"Link {link.attrib['name']} has no visual mesh")
    mesh_path = (urdf_path.parent / mesh.attrib["filename"]).resolve()
    triangles, _ = load_stl_triangles(str(mesh_path))
    scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ")
    visual_xyz, visual_rotation = element_transform(visual)
    triangles = triangles * scale
    triangles = triangles @ visual_rotation.T + visual_xyz
    triangles, normals = normals_from_triangles(triangles)
    return mesh_path, triangles, normals


def sample_camera_visible_surface(
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    *,
    camera_origin_surface: np.ndarray,
    camera_rotation_surface: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    sample_count: int,
    seed: int = 0,
    visibility_tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-sample the outer rubber surface and mark samples visible in the calibrated camera."""

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    triangle_normals = np.asarray(triangle_normals, dtype=np.float64).reshape(-1, 3)
    empty_points = np.empty((0, 3), dtype=np.float32)
    if int(sample_count) <= 0 or len(triangles) == 0:
        return empty_points, empty_points.copy(), np.empty(0, dtype=bool)

    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid_triangles = np.isfinite(areas) & (areas > 1.0e-18)
    if not np.any(valid_triangles):
        return empty_points, empty_points.copy(), np.empty(0, dtype=bool)

    rng = np.random.default_rng(int(seed))
    probabilities = np.where(valid_triangles, areas, 0.0)
    probabilities /= probabilities.sum()
    triangle_ids = rng.choice(len(triangles), int(sample_count), p=probabilities)
    root_u = np.sqrt(rng.random(int(sample_count)))
    v = rng.random(int(sample_count))
    barycentric = np.column_stack(
        (1.0 - root_u, root_u * (1.0 - v), root_u * v)
    )
    points = np.einsum("ni,nij->nj", barycentric, triangles[triangle_ids])
    normals = triangle_normals[triangle_ids].copy()

    camera_origin = np.asarray(camera_origin_surface, dtype=np.float64).reshape(3)
    camera_rotation = np.asarray(camera_rotation_surface, dtype=np.float64).reshape(3, 3)
    rays_surface = points - camera_origin
    ray_distance = np.linalg.norm(rays_surface, axis=1)
    ray_valid = np.isfinite(ray_distance) & (ray_distance > 1.0e-9)
    rays_surface[ray_valid] /= ray_distance[ray_valid, None]

    outer = np.zeros(len(points), dtype=bool)
    epsilon = 1.0e-5
    for index in np.flatnonzero(ray_valid):
        first_hit = ray_triangle_nearest(camera_origin, rays_surface[index], triangles)
        if first_hit is None:
            continue
        _, first_distance = first_hit
        second_origin = camera_origin + (first_distance + epsilon) * rays_surface[index]
        second_hit = ray_triangle_nearest(second_origin, rays_surface[index], triangles)
        outer_distance = (
            first_distance
            if second_hit is None
            else first_distance + epsilon + second_hit[1]
        )
        outer[index] = abs(outer_distance - ray_distance[index]) <= float(visibility_tolerance_m)

    points_camera = (points - camera_origin) @ camera_rotation
    in_front = points_camera[:, 2] > 1.0e-9
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[in_front] = distort_normalized(
        points_camera[in_front, :2] / points_camera[in_front, 2, None],
        distortion,
    )
    pixels[:, 0] = pixels[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2]
    pixels[:, 1] = pixels[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2]
    width, height = (int(value) for value in image_size)
    camera_visible = (
        outer
        & in_front
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )

    flip = np.einsum("ij,ij->i", normals, rays_surface) < 0.0
    normals[flip] *= -1.0
    return (
        points[outer].astype(np.float32),
        normals[outer].astype(np.float32),
        camera_visible[outer],
    )


def camera_visible_xy_hull(
    surface_points: np.ndarray,
    visible: np.ndarray,
    *,
    camera_origin_surface: np.ndarray,
    camera_rotation_surface: np.ndarray,
    include_camera_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Return the convex camera-XY boundary shared by ray generation and visualization."""

    points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
    visible_mask = np.asarray(visible, dtype=bool).reshape(-1)
    if len(points) != len(visible_mask):
        raise ValueError("Camera-visible point and mask counts do not match")
    camera_origin = np.asarray(camera_origin_surface, dtype=np.float64).reshape(3)
    camera_rotation = np.asarray(camera_rotation_surface, dtype=np.float64).reshape(3, 3)
    points_camera = (points - camera_origin) @ camera_rotation
    candidates = points_camera[visible_mask, :2]
    if include_camera_xy is not None:
        extra = np.asarray(include_camera_xy, dtype=np.float64).reshape(-1, 2)
        extra = extra[np.isfinite(extra).all(axis=1)]
        candidates = np.concatenate((candidates, extra), axis=0)
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    if len(candidates) < 3:
        raise ValueError("At least three visible camera-XY points are required")
    hull = cv2.convexHull(candidates.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError("Camera-visible XY hull is degenerate")
    return hull.astype(np.float32)


def clip_triangle_to_frustum_boundary_segments(
    triangle_camera: np.ndarray,
    clip_plane_normals: np.ndarray,
    *,
    boundary_plane_count: int,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Clip one triangle and return edges lying on camera-frustum side planes."""

    polygon = np.asarray(triangle_camera, dtype=np.float64).reshape(3, 3)
    planes = np.asarray(clip_plane_normals, dtype=np.float64).reshape(-1, 3)
    for plane in planes:
        if len(polygon) == 0:
            break
        distances = polygon @ plane
        clipped: list[np.ndarray] = []
        for index, current in enumerate(polygon):
            previous = polygon[index - 1]
            current_distance = float(distances[index])
            previous_distance = float(distances[index - 1])
            current_inside = current_distance >= -float(tolerance)
            previous_inside = previous_distance >= -float(tolerance)
            if current_inside != previous_inside:
                alpha = previous_distance / (previous_distance - current_distance)
                clipped.append(previous + alpha * (current - previous))
            if current_inside:
                clipped.append(current)
        polygon = np.asarray(clipped, dtype=np.float64).reshape(-1, 3)
    if len(polygon) < 2:
        return np.empty((0, 2, 3), dtype=np.float64)

    edges = np.stack((polygon, np.roll(polygon, -1, axis=0)), axis=1)
    side_distances = np.abs(edges @ planes[: int(boundary_plane_count)].T)
    on_side = np.any(np.all(side_distances <= float(tolerance), axis=1), axis=1)
    return edges[on_side]


def camera_frustum_surface_boundary_segments(
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
    *,
    camera_origin_surface: np.ndarray,
    camera_rotation_surface: np.ndarray,
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    surface_offset_m: float = 0.0,
    visibility_tolerance_m: float = 1.0e-4,
    ray_epsilon_m: float = 1.0e-5,
) -> np.ndarray:
    """Return the visible rubber/frustum intersection in the surface frame."""

    triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    triangle_normals = np.asarray(triangle_normals, dtype=np.float64).reshape(-1, 3)
    camera_origin = np.asarray(camera_origin_surface, dtype=np.float64).reshape(3)
    camera_rotation = np.asarray(camera_rotation_surface, dtype=np.float64).reshape(3, 3)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if len(triangles) != len(triangle_normals):
        raise ValueError("Triangle and normal counts do not match")

    width_px, height_px = (int(value) for value in image_size)
    x_min = -float(camera_matrix[0, 2]) / float(camera_matrix[0, 0])
    x_max = (float(width_px) - float(camera_matrix[0, 2])) / float(camera_matrix[0, 0])
    y_min = -float(camera_matrix[1, 2]) / float(camera_matrix[1, 1])
    y_max = (float(height_px) - float(camera_matrix[1, 2])) / float(camera_matrix[1, 1])
    clip_planes = np.asarray(
        (
            (1.0, 0.0, -x_min),
            (-1.0, 0.0, x_max),
            (0.0, 1.0, -y_min),
            (0.0, -1.0, y_max),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    triangles_camera = (triangles - camera_origin.reshape(1, 1, 3)) @ camera_rotation
    boundary_segments: list[np.ndarray] = []
    for triangle_camera, triangle_normal in zip(
        triangles_camera,
        triangle_normals,
        strict=True,
    ):
        segments_camera = clip_triangle_to_frustum_boundary_segments(
            triangle_camera,
            clip_planes,
            boundary_plane_count=4,
        )
        for segment_camera in segments_camera:
            segment_surface = segment_camera @ camera_rotation.T + camera_origin.reshape(1, 3)
            midpoint = np.mean(segment_surface, axis=0)
            ray = midpoint - camera_origin
            ray_distance = float(np.linalg.norm(ray))
            if not np.isfinite(ray_distance) or ray_distance <= 1.0e-9:
                continue
            ray /= ray_distance
            first_hit = ray_triangle_nearest(camera_origin, ray, triangles)
            if first_hit is None:
                continue
            outer_distance = float(first_hit[1])
            second_origin = camera_origin + (outer_distance + float(ray_epsilon_m)) * ray
            second_hit = ray_triangle_nearest(second_origin, ray, triangles)
            if second_hit is not None:
                outer_distance += float(ray_epsilon_m) + float(second_hit[1])
            if abs(outer_distance - ray_distance) > float(visibility_tolerance_m):
                continue

            normal = np.asarray(triangle_normal, dtype=np.float64).copy()
            if float(np.dot(normal, ray)) < 0.0:
                normal *= -1.0
            boundary_segments.append(
                segment_surface + normal.reshape(1, 3) * float(surface_offset_m)
            )

    if not boundary_segments:
        raise ValueError("Camera frustum does not intersect the visible rubber surface")
    return np.asarray(boundary_segments, dtype=np.float32).reshape(-1, 2, 3)


def order_closed_boundary_segments_xy(
    boundary_segments_xy: np.ndarray,
    *,
    join_tolerance_m: float = 2.0e-6,
) -> np.ndarray:
    """Order an unordered, single-cycle XY segment set into a closed polygon."""

    segments = np.asarray(boundary_segments_xy, dtype=np.float64).reshape(-1, 2, 2)
    if len(segments) < 3 or not np.isfinite(segments).all():
        raise ValueError("Camera-range boundary needs at least three finite segments")
    if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1) <= 1.0e-12):
        raise ValueError("Camera-range boundary contains a degenerate segment")

    used = np.zeros(len(segments), dtype=bool)
    used[0] = True
    ordered = [segments[0, 0], segments[0, 1]]
    for _ in range(1, len(segments)):
        remaining = np.flatnonzero(~used)
        endpoint_distances = np.linalg.norm(
            segments[remaining] - ordered[-1].reshape(1, 1, 2),
            axis=2,
        )
        flat_index = int(np.argmin(endpoint_distances))
        distance = float(endpoint_distances.reshape(-1)[flat_index])
        if distance > float(join_tolerance_m):
            raise ValueError(
                "Camera-range boundary is not one closed cycle: "
                f"nearest endpoint gap is {distance:.3e} m"
            )
        remaining_offset, matched_endpoint = np.unravel_index(
            flat_index,
            endpoint_distances.shape,
        )
        segment_index = int(remaining[remaining_offset])
        used[segment_index] = True
        ordered.append(segments[segment_index, 1 - int(matched_endpoint)])

    polygon = np.asarray(ordered, dtype=np.float64)
    closure_error = float(np.linalg.norm(polygon[-1] - polygon[0]))
    if closure_error > float(join_tolerance_m):
        raise ValueError(
            "Camera-range boundary is open: "
            f"closure error is {closure_error:.3e} m"
        )
    polygon = polygon[:-1]
    signed_area_twice = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
    )
    if abs(signed_area_twice) <= 1.0e-12:
        raise ValueError("Camera-range boundary polygon has zero area")
    if signed_area_twice < 0.0:
        polygon = polygon[::-1]
    first = int(np.lexsort((polygon[:, 0], polygon[:, 1]))[0])
    polygon = np.roll(polygon, -first, axis=0)
    return polygon.astype(np.float32)


def compose_camera_plane_ray_grid(
    hull_xy_camera: np.ndarray,
    *,
    camera_origin_link: np.ndarray,
    camera_rotation_link: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a structured base ray grid inside one camera-XY boundary."""

    hull = np.asarray(hull_xy_camera, dtype=np.float64).reshape(-1, 2)
    origin = np.asarray(camera_origin_link, dtype=np.float64).reshape(3)
    rotation = np.asarray(camera_rotation_link, dtype=np.float64).reshape(3, 3)
    rows = int(rows)
    cols = int(cols)
    if len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError("Camera-visible XY hull must contain at least three finite points")
    if rows <= 0 or cols <= 0:
        raise ValueError("Ray-grid dimensions must be positive")
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
        raise ValueError("Camera extrinsics contain non-finite values")

    # Keep a monotonic 32x24 topology. Each row is a camera-Y scanline and
    # every column stays at its regular cell-center position inside that row.
    y_min = float(np.min(hull[:, 1]))
    y_max = float(np.max(hull[:, 1]))
    if y_max - y_min <= 1.0e-9:
        raise ValueError("Camera-visible XY hull has no Y extent")
    xy_grid = np.empty((rows, cols, 2), dtype=np.float64)
    hull_next = np.roll(hull, -1, axis=0)
    for row in range(rows):
        y = y_min + (row + 0.5) / rows * (y_max - y_min)
        intersections: list[float] = []
        for start, end in zip(hull, hull_next, strict=True):
            y0 = float(start[1])
            y1 = float(end[1])
            if (y0 <= y < y1) or (y1 <= y < y0):
                intersections.append(
                    float(start[0] + (y - y0) * (end[0] - start[0]) / (y1 - y0))
                )
        intersections.sort()
        unique_intersections = [
            value
            for index, value in enumerate(intersections)
            if index == 0 or abs(value - intersections[index - 1]) > 1.0e-9
        ]
        if len(unique_intersections) != 2:
            raise ValueError(
                "Camera-XY boundary must have one continuous scanline span; "
                f"row {row} has {len(unique_intersections)} intersections"
            )
        x_min, x_max = unique_intersections
        for col in range(cols):
            xy_grid[row, col] = (
                x_min + (col + 0.5) / cols * (x_max - x_min),
                y,
            )

    starts_camera = np.concatenate(
        (xy_grid, np.zeros((rows, cols, 1), dtype=np.float64)),
        axis=-1,
    )
    starts_link = starts_camera @ rotation.T + origin.reshape(1, 1, 3)
    optical_axis_link = rotation[:, 2]
    optical_axis_norm = float(np.linalg.norm(optical_axis_link))
    if optical_axis_norm <= 1.0e-9:
        raise ValueError("Camera optical axis is degenerate")
    optical_axis_link = optical_axis_link / optical_axis_norm
    directions_link = np.broadcast_to(optical_axis_link, starts_link.shape).copy()
    return starts_link.astype(np.float32), directions_link.astype(np.float32)


def camera_xy_to_ray_grid_coordinates(
    hull_xy_camera: np.ndarray,
    points_xy_camera: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera-XY points into the continuous structured ray-grid coordinates.

    The returned coordinates use ``(row, col)`` order and the same cell-center
    convention as :func:`compose_camera_plane_ray_grid`: the first cell center
    is ``0`` and the outer hull edges are ``-0.5``/``size - 0.5``.
    """

    hull = np.asarray(hull_xy_camera, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(points_xy_camera, dtype=np.float64).reshape(-1, 2)
    rows = int(rows)
    cols = int(cols)
    if len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError("Camera-visible XY hull must contain at least three finite points")
    if rows <= 0 or cols <= 0:
        raise ValueError("Ray-grid dimensions must be positive")

    y_min = float(np.min(hull[:, 1]))
    y_max = float(np.max(hull[:, 1]))
    y_extent = y_max - y_min
    if y_extent <= 1.0e-9:
        raise ValueError("Camera-visible XY hull has no Y extent")

    coordinates = np.full((len(points), 2), np.nan, dtype=np.float64)
    inside = np.zeros(len(points), dtype=bool)
    hull_next = np.roll(hull, -1, axis=0)
    tolerance = max(1.0e-9, y_extent * 1.0e-7)
    for point_index, (x, y) in enumerate(points):
        if not np.isfinite((x, y)).all() or y < y_min - tolerance or y > y_max + tolerance:
            continue
        y_eval = float(np.clip(y, y_min, y_max))
        intersections: list[float] = []
        for start, end in zip(hull, hull_next, strict=True):
            x0, y0 = (float(value) for value in start)
            x1, y1 = (float(value) for value in end)
            if abs(y1 - y0) <= tolerance:
                if abs(y_eval - y0) <= tolerance:
                    intersections.extend((x0, x1))
                continue
            if min(y0, y1) - tolerance <= y_eval <= max(y0, y1) + tolerance:
                alpha = np.clip((y_eval - y0) / (y1 - y0), 0.0, 1.0)
                intersections.append(float(x0 + alpha * (x1 - x0)))
        if len(intersections) < 2:
            continue
        x_min = min(intersections)
        x_max = max(intersections)
        x_extent = x_max - x_min
        if x < x_min - tolerance or x > x_max + tolerance:
            continue
        row = (y_eval - y_min) / y_extent * rows - 0.5
        col = (
            0.5 * (cols - 1)
            if x_extent <= tolerance
            else (float(np.clip(x, x_min, x_max)) - x_min) / x_extent * cols - 0.5
        )
        coordinates[point_index] = (row, col)
        inside[point_index] = True

    return coordinates.astype(np.float32), inside


def compose_exact_marker_camera_plane_rays(
    marker_points_camera: np.ndarray,
    marker_valid: np.ndarray,
    *,
    camera_origin_link: np.ndarray,
    camera_rotation_link: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one independent camera-Z ray slot for every calibrated marker."""

    marker_points = np.asarray(marker_points_camera, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(marker_valid, dtype=bool).reshape(-1)
    origin = np.asarray(camera_origin_link, dtype=np.float64).reshape(3)
    rotation = np.asarray(camera_rotation_link, dtype=np.float64).reshape(3, 3)
    if len(marker_points) != len(valid):
        raise ValueError("Marker camera point and validity counts do not match")
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
        raise ValueError("Camera extrinsics contain non-finite values")
    if np.any(valid & ~np.isfinite(marker_points).all(axis=1)):
        raise ValueError("Valid marker rays contain non-finite camera points")

    # Invalid marker IDs retain finite, harmless slots so every finger keeps
    # the fixed 10x10 HydroShear marker tensor shape.
    starts_camera = np.zeros((len(marker_points), 3), dtype=np.float64)
    starts_camera[valid, :2] = marker_points[valid, :2]
    starts_link = starts_camera @ rotation.T + origin.reshape(1, 3)
    optical_axis_link = rotation[:, 2]
    optical_axis_norm = float(np.linalg.norm(optical_axis_link))
    if optical_axis_norm <= 1.0e-9:
        raise ValueError("Camera optical axis is degenerate")
    optical_axis_link = optical_axis_link / optical_axis_norm
    directions_link = np.broadcast_to(optical_axis_link, starts_link.shape).copy()
    return (
        starts_link.astype(np.float32),
        directions_link.astype(np.float32),
        valid.copy(),
    )


def compose_marker_priority_camera_plane_ray_grid(
    hull_xy_camera: np.ndarray,
    marker_points_camera: np.ndarray,
    marker_pixels: np.ndarray,
    marker_valid: np.ndarray,
    *,
    camera_origin_link: np.ndarray,
    camera_rotation_link: np.ndarray,
    rows: int,
    cols: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a structured camera-XY ray grid with exact marker cells first."""

    hull = np.asarray(hull_xy_camera, dtype=np.float64).reshape(-1, 2)
    marker_points = np.asarray(marker_points_camera, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(marker_pixels, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(marker_valid, dtype=bool).reshape(-1)
    origin = np.asarray(camera_origin_link, dtype=np.float64).reshape(3)
    rotation = np.asarray(camera_rotation_link, dtype=np.float64).reshape(3, 3)
    rows = int(rows)
    cols = int(cols)
    width, height = (int(value) for value in image_size)
    if len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError("Camera-visible XY hull must contain at least three finite points")
    if not (len(marker_points) == len(pixels) == len(valid)):
        raise ValueError("Marker camera point, pixel, and valid counts do not match")
    if rows <= 0 or cols <= 0 or width <= 0 or height <= 0:
        raise ValueError("Ray-grid and camera-image dimensions must be positive")
    if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
        raise ValueError("Camera extrinsics contain non-finite values")

    # Preserve the 32x24 spatial topology. Each row is a camera-Y scanline;
    # its columns are distributed between the two convex-hull intersections.
    y_min = float(np.min(hull[:, 1]))
    y_max = float(np.max(hull[:, 1]))
    if y_max - y_min <= 1.0e-9:
        raise ValueError("Camera-visible XY hull has no Y extent")
    xy_grid = np.empty((rows, cols, 2), dtype=np.float64)
    hull_next = np.roll(hull, -1, axis=0)
    for row in range(rows):
        y = y_min + (row + 0.5) / rows * (y_max - y_min)
        intersections: list[float] = []
        for start, end in zip(hull, hull_next, strict=True):
            y0 = float(start[1])
            y1 = float(end[1])
            if (y0 <= y < y1) or (y1 <= y < y0):
                intersections.append(
                    float(start[0] + (y - y0) * (end[0] - start[0]) / (y1 - y0))
                )
        if len(intersections) < 2:
            raise ValueError(f"Camera-visible hull has no scanline span for row {row}")
        x_min = min(intersections)
        x_max = max(intersections)
        for col in range(cols):
            xy_grid[row, col] = (
                x_min + (col + 0.5) / cols * (x_max - x_min),
                y,
            )

    # HydroShear samples marker depth through the marker's calibrated image UV.
    # Reserve that exact 32x24 cell and replace its regular XY with the marker's
    # metric camera XY, so the sampled depth comes from the marker ray itself.
    marker_mask = np.zeros((rows, cols), dtype=bool)
    valid_ids = np.flatnonzero(valid)
    marker_cols = np.clip(np.rint(pixels[valid_ids, 0] * cols / width).astype(np.int64), 0, cols - 1)
    marker_rows = np.clip(np.rint(pixels[valid_ids, 1] * rows / height).astype(np.int64), 0, rows - 1)
    marker_flat_ids = marker_rows * cols + marker_cols
    if len(np.unique(marker_flat_ids)) != len(marker_flat_ids):
        raise ValueError("Calibrated markers map to the same ray-grid cell")
    xy_grid.reshape(-1, 2)[marker_flat_ids] = marker_points[valid_ids, :2]
    marker_mask.reshape(-1)[marker_flat_ids] = True

    starts_camera = np.concatenate(
        (xy_grid, np.zeros((rows, cols, 1), dtype=np.float64)),
        axis=-1,
    )
    starts_link = starts_camera @ rotation.T + origin.reshape(1, 1, 3)
    optical_axis_link = rotation[:, 2]
    optical_axis_norm = float(np.linalg.norm(optical_axis_link))
    if optical_axis_norm <= 1.0e-9:
        raise ValueError("Camera optical axis is degenerate")
    optical_axis_link = optical_axis_link / optical_axis_norm
    directions_link = np.broadcast_to(optical_axis_link, starts_link.shape).copy()
    return (
        starts_link.astype(np.float32),
        directions_link.astype(np.float32),
        marker_mask,
    )


def marker_surface_hit_index_grid(
    ray_starts_link: np.ndarray,
    ray_directions_link: np.ndarray,
    marker_points_link: np.ndarray,
    marker_pixels: np.ndarray,
    marker_valid: np.ndarray,
    triangles: np.ndarray,
    *,
    rows: int,
    cols: int,
    image_size: tuple[int, int],
    second_hit_epsilon_m: float = 1.0e-5,
) -> np.ndarray:
    """Select the first/second rubber hit that reproduces each calibrated marker."""

    starts = np.asarray(ray_starts_link, dtype=np.float64).reshape(rows * cols, 3)
    directions = np.asarray(ray_directions_link, dtype=np.float64).reshape(rows * cols, 3)
    marker_points = np.asarray(marker_points_link, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(marker_pixels, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(marker_valid, dtype=bool).reshape(-1)
    width, height = (int(value) for value in image_size)
    hit_indices = np.full((rows, cols), 2, dtype=np.uint8)
    valid_ids = np.flatnonzero(valid)
    marker_cols = np.clip(
        np.rint(pixels[valid_ids, 0] * cols / width).astype(np.int64),
        0,
        cols - 1,
    )
    marker_rows = np.clip(
        np.rint(pixels[valid_ids, 1] * rows / height).astype(np.int64),
        0,
        rows - 1,
    )
    epsilon = max(0.0, float(second_hit_epsilon_m))
    for marker_id, row, col in zip(
        valid_ids,
        marker_rows,
        marker_cols,
        strict=True,
    ):
        ray_id = int(row) * cols + int(col)
        start = starts[ray_id]
        direction = directions[ray_id]
        target_depth = float(np.dot(marker_points[marker_id] - start, direction))
        first_hit = ray_triangle_nearest(start, direction, triangles)
        if first_hit is None:
            raise ValueError(f"Marker {marker_id} camera-plane ray does not hit the rubber mesh")
        _, first_depth = first_hit
        candidates = [(1, float(first_depth))]
        second_start = start + (float(first_depth) + epsilon) * direction
        second_hit = ray_triangle_nearest(second_start, direction, triangles)
        if second_hit is not None:
            _, second_depth = second_hit
            candidates.append(
                (2, float(first_depth) + epsilon + float(second_depth))
            )
        selected_index, selected_depth = min(
            candidates,
            key=lambda item: abs(item[1] - target_depth),
        )
        if abs(selected_depth - target_depth) > 1.0e-5:
            raise ValueError(
                f"Marker {marker_id} rubber hit misses calibration by "
                f"{abs(selected_depth - target_depth) * 1.0e6:.3f}um"
            )
        hit_indices[int(row), int(col)] = selected_index
    return hit_indices


def marker_surface_hit_indices(
    ray_starts_link: np.ndarray,
    ray_directions_link: np.ndarray,
    marker_points_link: np.ndarray,
    marker_valid: np.ndarray,
    triangles: np.ndarray,
    *,
    second_hit_epsilon_m: float = 1.0e-5,
) -> np.ndarray:
    """Select the calibrated rubber intersection for independent marker rays."""

    starts = np.asarray(ray_starts_link, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(ray_directions_link, dtype=np.float64).reshape(-1, 3)
    marker_points = np.asarray(marker_points_link, dtype=np.float64).reshape(-1, 3)
    valid = np.asarray(marker_valid, dtype=bool).reshape(-1)
    if not (len(starts) == len(directions) == len(marker_points) == len(valid)):
        raise ValueError("Independent marker ray arrays must have equal lengths")

    hit_indices = np.full(len(starts), 2, dtype=np.uint8)
    epsilon = max(0.0, float(second_hit_epsilon_m))
    for marker_id in np.flatnonzero(valid):
        start = starts[marker_id]
        direction = directions[marker_id]
        target_depth = float(np.dot(marker_points[marker_id] - start, direction))
        first_hit = ray_triangle_nearest(start, direction, triangles)
        if first_hit is None:
            raise ValueError(f"Marker {marker_id} camera-plane ray does not hit the rubber mesh")
        _, first_depth = first_hit
        candidates = [(1, float(first_depth))]
        second_start = start + (float(first_depth) + epsilon) * direction
        second_hit = ray_triangle_nearest(second_start, direction, triangles)
        if second_hit is not None:
            _, second_depth = second_hit
            candidates.append((2, float(first_depth) + epsilon + float(second_depth)))
        selected_index, selected_depth = min(
            candidates,
            key=lambda item: abs(item[1] - target_depth),
        )
        if abs(selected_depth - target_depth) > 1.0e-5:
            raise ValueError(
                f"Marker {marker_id} rubber hit misses calibration by "
                f"{abs(selected_depth - target_depth) * 1.0e6:.3f}um"
            )
        hit_indices[marker_id] = selected_index
    return hit_indices


def outer_surface_hit_indices(
    ray_starts: np.ndarray,
    ray_directions: np.ndarray,
    triangles: np.ndarray,
    *,
    ray_epsilon_m: float = 1.0e-5,
) -> np.ndarray:
    """Select the farthest available rubber hit for every structured ray."""

    starts = np.asarray(ray_starts, dtype=np.float64)
    directions = np.asarray(ray_directions, dtype=np.float64)
    if starts.shape != directions.shape or starts.shape[-1] != 3:
        raise ValueError("Structured ray starts and directions must have matching (..., 3) shapes")
    flat_starts = starts.reshape(-1, 3)
    flat_directions = directions.reshape(-1, 3)
    hit_indices = np.empty(len(flat_starts), dtype=np.uint8)
    epsilon = max(0.0, float(ray_epsilon_m))
    for index, (ray_origin, ray_direction) in enumerate(
        zip(flat_starts, flat_directions, strict=True)
    ):
        first_hit = ray_triangle_nearest(ray_origin, ray_direction, triangles)
        if first_hit is None:
            raise ValueError(f"Structured camera ray {index} misses the rubber surface")
        second_origin = ray_origin + (float(first_hit[1]) + epsilon) * ray_direction
        second_hit = ray_triangle_nearest(second_origin, ray_direction, triangles)
        hit_indices[index] = 1 if second_hit is None else 2
    return hit_indices.reshape(starts.shape[:-1])


def closest_mesh_point_to_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
) -> tuple[int, np.ndarray, float]:
    """Return the mesh boundary point nearest to a forward ray."""

    edges = np.stack(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=1,
    ).reshape(-1, 2, 3)
    triangle_ids = np.repeat(np.arange(len(triangles)), 3)
    starts = edges[:, 0]
    vectors = edges[:, 1] - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    valid_edges = lengths_sq > 1.0e-18
    best_distance_sq = np.inf
    best_point = None
    best_edge = -1

    def consider(points: np.ndarray, ray_t: np.ndarray, valid: np.ndarray) -> None:
        nonlocal best_distance_sq, best_point, best_edge
        ray_points = origin + ray_t[:, None] * direction
        distance_sq = np.einsum("ij,ij->i", points - ray_points, points - ray_points)
        distance_sq = np.where(valid, distance_sq, np.inf)
        edge_index = int(np.argmin(distance_sq))
        if distance_sq[edge_index] < best_distance_sq:
            best_distance_sq = float(distance_sq[edge_index])
            best_point = points[edge_index].copy()
            best_edge = edge_index

    offset = origin - starts
    ray_dot_edge = vectors @ direction
    ray_dot_offset = offset @ direction
    edge_dot_offset = np.einsum("ij,ij->i", vectors, offset)
    denominator = lengths_sq - ray_dot_edge * ray_dot_edge
    valid_interior = valid_edges & (np.abs(denominator) > 1.0e-18)
    ray_t = np.zeros(len(edges), dtype=np.float64)
    edge_u = np.zeros(len(edges), dtype=np.float64)
    ray_t[valid_interior] = (
        ray_dot_edge[valid_interior] * edge_dot_offset[valid_interior]
        - lengths_sq[valid_interior] * ray_dot_offset[valid_interior]
    ) / denominator[valid_interior]
    edge_u[valid_interior] = (
        edge_dot_offset[valid_interior]
        - ray_dot_edge[valid_interior] * ray_dot_offset[valid_interior]
    ) / denominator[valid_interior]
    valid_interior &= (ray_t >= 0.0) & (edge_u >= 0.0) & (edge_u <= 1.0)
    consider(starts + edge_u[:, None] * vectors, ray_t, valid_interior)

    for points in (edges[:, 0], edges[:, 1]):
        endpoint_t = np.maximum((points - origin) @ direction, 0.0)
        consider(points, endpoint_t, valid_edges)

    origin_u = np.zeros(len(edges), dtype=np.float64)
    origin_u[valid_edges] = np.clip(
        np.einsum("ij,ij->i", origin - starts, vectors)[valid_edges]
        / lengths_sq[valid_edges],
        0.0,
        1.0,
    )
    consider(
        starts + origin_u[:, None] * vectors,
        np.zeros(len(edges), dtype=np.float64),
        valid_edges,
    )

    if best_point is None:
        raise RuntimeError("Surface mesh has no non-degenerate edges")
    return int(triangle_ids[best_edge]), best_point, float(np.sqrt(best_distance_sq))


def project_point(
    point_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    monotonic_radial_scale: float | None = None,
) -> np.ndarray:
    if monotonic_radial_scale is not None:
        normalized = point_camera[:2] / point_camera[2]
        distorted = distort_normalized_monotonic(
            normalized,
            distortion,
            monotonic_radial_scale,
        )[0]
        return np.array(
            (
                distorted[0] * camera_matrix[0, 0] + camera_matrix[0, 2],
                distorted[1] * camera_matrix[1, 1] + camera_matrix[1, 2],
            )
        )
    return cv2.projectPoints(
        point_camera.reshape(1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion,
    )[0].reshape(2)


def project_finger(
    *,
    marker_ids: np.ndarray,
    pixels: np.ndarray,
    rays_camera: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    distortion_valid: np.ndarray,
    monotonic_radial_scale: float,
    camera_origin_surface: np.ndarray,
    camera_rotation_surface: np.ndarray,
    triangles: np.ndarray,
    triangle_normals: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(marker_ids)
    points_surface = np.empty((count, 3), dtype=np.float64)
    normals_surface = np.empty((count, 3), dtype=np.float64)
    points_camera = np.empty((count, 3), dtype=np.float64)
    distance_m = np.empty(count, dtype=np.float64)
    surface_error_m = np.empty(count, dtype=np.float64)
    mesh_reprojection_error_px = np.empty(count, dtype=np.float64)
    triangle_ids = np.empty(count, dtype=np.int32)
    methods = np.empty(count, dtype="<U16")

    for index, (pixel, ray_camera) in enumerate(zip(pixels, rays_camera)):
        ray_surface = camera_rotation_surface @ ray_camera
        hit = ray_triangle_nearest(camera_origin_surface, ray_surface, triangles)
        if hit is None:
            triangle_id, point_surface, surface_error = closest_mesh_point_to_ray(
                camera_origin_surface,
                ray_surface,
                triangles,
            )
            method = "nearest_surface"
        else:
            triangle_id, ray_distance = hit
            outer_origin = camera_origin_surface + (ray_distance + 1.0e-5) * ray_surface
            outer_hit = ray_triangle_nearest(outer_origin, ray_surface, triangles)
            if outer_hit is not None:
                triangle_id, outer_distance = outer_hit
                ray_distance += 1.0e-5 + outer_distance
            point_surface = camera_origin_surface + ray_distance * ray_surface
            surface_error = 0.0
            method = "ray_hit"

        normal_surface = triangle_normals[triangle_id].copy()
        if float(np.dot(normal_surface, point_surface - camera_origin_surface)) < 0.0:
            normal_surface *= -1.0
        point_camera = camera_rotation_surface.T @ (
            point_surface - camera_origin_surface
        )
        reprojected = project_point(
            point_camera,
            camera_matrix,
            distortion,
            None if distortion_valid[index] else monotonic_radial_scale,
        )

        points_surface[index] = point_surface
        normals_surface[index] = normal_surface
        points_camera[index] = point_camera
        distance_m[index] = np.linalg.norm(point_camera)
        surface_error_m[index] = surface_error
        mesh_reprojection_error_px[index] = np.linalg.norm(reprojected - pixel)
        triangle_ids[index] = triangle_id
        methods[index] = method

    return {
        "points_link_m": points_surface.astype(np.float32),
        "normals_link": normals_surface.astype(np.float32),
        "points_camera_m": points_camera.astype(np.float32),
        "distance_m": distance_m.astype(np.float32),
        "surface_error_m": surface_error_m.astype(np.float32),
        "mesh_reprojection_error_px": mesh_reprojection_error_px.astype(np.float32),
        "triangle_ids": triangle_ids,
        "method": methods,
    }


def main() -> None:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markers",
        type=Path,
        default=repo_root
        / "vitai_4Fingers-320*240"
        / "marker_annotations"
        / "marker_centers.csv",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=repo_root
        / "vitai_4Fingers-320*240"
        / "camera_intrinsics_320x240.yaml",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=repo_root
        / "assets"
        / "revo21_right_touch"
        / "urdf"
        / "revo21_dv2_urdf_right-touch.SLDASM.urdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "assets"
        / "revo21_right_touch"
        / "marker_positions"
        / "vitai_4fingers",
    )
    args = parser.parse_args()

    camera_matrix, distortion, image_size = load_intrinsics(args.intrinsics)
    marker_ids, pixels = load_marker_pixels(args.markers)
    if np.any(pixels < 0.0) or np.any(pixels >= np.asarray(image_size)):
        raise ValueError("Marker pixels fall outside the calibrated image")
    (
        normalized,
        brown_reprojection_error_px,
        selected_model_reprojection_error_px,
        distortion_valid,
        monotonic_radial_scale,
        monotonic_fit_radius,
        monotonic_fit_rms,
    ) = invert_distortion(pixels, camera_matrix, distortion)
    distortion_model = np.where(
        distortion_valid,
        "brown_exact",
        "monotonic_extrapolation",
    )
    undistorted_pixels = np.column_stack(
        (
            normalized[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2],
            normalized[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2],
        )
    )
    rays_camera = np.column_stack((normalized, np.ones(len(normalized))))
    rays_camera /= np.linalg.norm(rays_camera, axis=1, keepdims=True)

    urdf_path = args.urdf.resolve()
    root = ET.parse(urdf_path).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    output: dict[str, np.ndarray] = {
        "marker_ids": marker_ids,
        "pixels_distorted": pixels.astype(np.float32),
        "pixels_undistorted": undistorted_pixels.astype(np.float32),
        "ray_reprojection_error_px": brown_reprojection_error_px.astype(np.float32),
        "brown_reprojection_error_px": brown_reprojection_error_px.astype(np.float32),
        "selected_model_reprojection_error_px": (
            selected_model_reprojection_error_px.astype(np.float32)
        ),
        "distortion_valid": distortion_valid,
        "distortion_model": distortion_model,
    }
    csv_rows = []
    summary = {
        "markers": str(args.markers.resolve()),
        "intrinsics": str(args.intrinsics.resolve()),
        "urdf": str(urdf_path),
        "image_size": list(image_size),
        "marker_count": int(len(marker_ids)),
        "surface": "fingertip rubber visual mesh",
        "camera_convention": "OpenCV: +x right, +y down, +z forward",
        "distortion_inverse": {
            "model": "Brown-Conrady exact branch with monotonic atan extrapolation",
            "max_valid_reprojection_error_px": 0.5,
            "valid_marker_count": int(np.count_nonzero(distortion_valid)),
            "extrapolated_marker_count": int(np.count_nonzero(~distortion_valid)),
            "extrapolated_marker_ids": marker_ids[~distortion_valid].tolist(),
            "monotonic_radial_scale": monotonic_radial_scale,
            "fit_radius": monotonic_fit_radius,
            "fit_rms_px": monotonic_fit_rms
            * float(np.sqrt(camera_matrix[0, 0] * camera_matrix[1, 1])),
        },
        "fingers": {},
    }

    for finger, config in FINGERS.items():
        camera_joint = joints[config["camera_joint"]]
        surface_joint = joints[config["surface_joint"]]
        surface_link = links[config["surface_link"]]
        if camera_joint.find("parent").attrib["link"] != surface_joint.find("parent").attrib["link"]:
            raise ValueError(f"{finger} camera and rubber surface do not share a parent link")

        camera_xyz_parent, camera_rotation_parent = element_transform(camera_joint)
        surface_xyz_parent, surface_rotation_parent = element_transform(surface_joint)
        camera_origin_surface = surface_rotation_parent.T @ (
            camera_xyz_parent - surface_xyz_parent
        )
        camera_rotation_surface = (
            surface_rotation_parent.T @ camera_rotation_parent
        )
        mesh_path, triangles, triangle_normals = load_surface_triangles(
            urdf_path,
            surface_link,
        )
        result = project_finger(
            marker_ids=marker_ids,
            pixels=pixels,
            rays_camera=rays_camera,
            camera_matrix=camera_matrix,
            distortion=distortion,
            distortion_valid=distortion_valid,
            monotonic_radial_scale=monotonic_radial_scale,
            camera_origin_surface=camera_origin_surface,
            camera_rotation_surface=camera_rotation_surface,
            triangles=triangles,
            triangle_normals=triangle_normals,
        )
        for key, value in result.items():
            output[f"{finger}_{key}"] = value
        output[f"{finger}_camera_origin_link_m"] = camera_origin_surface.astype(
            np.float32
        )
        output[f"{finger}_camera_rotation_link"] = camera_rotation_surface.astype(
            np.float32
        )

        surface_boundary_segments = camera_frustum_surface_boundary_segments(
            triangles,
            triangle_normals,
            camera_origin_surface=camera_origin_surface,
            camera_rotation_surface=camera_rotation_surface,
            camera_matrix=camera_matrix,
            image_size=image_size,
        )
        surface_boundary_camera = (
            surface_boundary_segments - camera_origin_surface.reshape(1, 1, 3)
        ) @ camera_rotation_surface
        camera_range_boundary_xy = order_closed_boundary_segments_xy(
            surface_boundary_camera[..., :2]
        )
        ray_starts, ray_directions = compose_camera_plane_ray_grid(
            camera_range_boundary_xy,
            camera_origin_link=camera_origin_surface,
            camera_rotation_link=camera_rotation_surface,
            rows=RAY_ROWS,
            cols=RAY_COLS,
        )
        ray_is_marker = np.zeros((RAY_ROWS, RAY_COLS), dtype=bool)
        ray_surface_hit_index = outer_surface_hit_indices(
            ray_starts,
            ray_directions,
            triangles,
        )
        marker_exact = (
            distortion_valid
            & (result["method"] == "ray_hit")
            & np.isfinite(result["points_camera_m"]).all(axis=1)
        )
        marker_ray_starts, marker_ray_directions, marker_ray_valid = (
            compose_exact_marker_camera_plane_rays(
                result["points_camera_m"],
                marker_exact,
                camera_origin_link=camera_origin_surface,
                camera_rotation_link=camera_rotation_surface,
            )
        )
        marker_ray_surface_hit_index = marker_surface_hit_indices(
            marker_ray_starts,
            marker_ray_directions,
            result["points_link_m"],
            marker_ray_valid,
            triangles,
        )
        # Preserve the legacy key for existing consumers while recording the
        # precise source of the new 32x24 layout under an explicit key.
        output[f"{finger}_camera_visible_hull_xy_m"] = camera_range_boundary_xy
        output[f"{finger}_camera_range_aug4_boundary_xy_m"] = camera_range_boundary_xy
        output[f"{finger}_ray_starts_link_m"] = ray_starts
        output[f"{finger}_ray_directions_link"] = ray_directions
        output[f"{finger}_ray_is_marker"] = ray_is_marker
        output[f"{finger}_ray_surface_hit_index"] = ray_surface_hit_index
        output[f"{finger}_marker_ray_starts_link_m"] = marker_ray_starts
        output[f"{finger}_marker_ray_directions_link"] = marker_ray_directions
        output[f"{finger}_marker_ray_valid"] = marker_ray_valid
        output[f"{finger}_marker_ray_surface_hit_index"] = marker_ray_surface_hit_index

        exact_hits = int(np.count_nonzero(result["method"] == "ray_hit"))
        fallback_count = int(len(marker_ids) - exact_hits)
        summary["fingers"][finger] = {
            "camera_joint": config["camera_joint"],
            "surface_link": config["surface_link"],
            "surface_mesh": str(mesh_path),
            "exact_ray_hits": exact_hits,
            "nearest_surface_fallbacks": fallback_count,
            "distance_m": {
                "min": float(np.min(result["distance_m"])),
                "median": float(np.median(result["distance_m"])),
                "max": float(np.max(result["distance_m"])),
            },
            "max_reprojection_error_px": float(
                np.max(result["mesh_reprojection_error_px"])
            ),
            "camera_plane_ray_layout": {
                "shape": [RAY_ROWS, RAY_COLS],
                "marker_overridden_base_ray_count": 0,
                "marker_ray_shape": [len(marker_ids)],
                "marker_ray_count": int(np.count_nonzero(marker_ray_valid)),
                "marker_first_hit_count": int(
                    np.count_nonzero(marker_ray_valid & (marker_ray_surface_hit_index == 1))
                ),
                "marker_second_hit_count": int(
                    np.count_nonzero(marker_ray_valid & (marker_ray_surface_hit_index == 2))
                ),
                "base_first_hit_count": int(np.count_nonzero(ray_surface_hit_index == 1)),
                "base_second_hit_count": int(np.count_nonzero(ray_surface_hit_index == 2)),
                "origin_camera_z_m": 0.0,
                "direction_camera": [0.0, 0.0, 1.0],
                "sampling_boundary": "camera_range_aug4",
                "camera_range_boundary_segment_count": int(len(surface_boundary_segments)),
                "camera_range_boundary_vertex_count": int(len(camera_range_boundary_xy)),
            },
        }
        for index, marker_id in enumerate(marker_ids):
            point_link = result["points_link_m"][index]
            point_camera = result["points_camera_m"][index]
            normal_link = result["normals_link"][index]
            csv_rows.append(
                [
                    finger,
                    marker_id,
                    *pixels[index],
                    *undistorted_pixels[index],
                    *point_camera,
                    *point_link,
                    *normal_link,
                    result["distance_m"][index],
                    distortion_model[index],
                    brown_reprojection_error_px[index],
                    selected_model_reprojection_error_px[index],
                    result["method"][index],
                    result["surface_error_m"][index],
                    result["mesh_reprojection_error_px"][index],
                    config["surface_link"],
                    int(result["triangle_ids"][index]),
                ]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "marker_positions.npz", **output)
    with (args.output_dir / "marker_positions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "finger",
                "marker_id",
                "u_px",
                "v_px",
                "u_undistorted_px",
                "v_undistorted_px",
                "x_camera_m",
                "y_camera_m",
                "z_camera_m",
                "x_link_m",
                "y_link_m",
                "z_link_m",
                "nx_link",
                "ny_link",
                "nz_link",
                "distance_m",
                "distortion_inverse",
                "brown_reprojection_error_px",
                "selected_model_reprojection_error_px",
                "method",
                "surface_error_m",
                "mesh_reprojection_error_px",
                "surface_link",
                "triangle_id",
            ]
        )
        writer.writerows(csv_rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "distortion_inverse": summary["distortion_inverse"],
                "fingers": summary["fingers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
