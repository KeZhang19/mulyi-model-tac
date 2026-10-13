"""Filter actual SDK marker components in a calibrated image coordinate space.

This module opens no hardware and does not alter or crop RGB images. Pass its
detector callback to MarkerTracking's ``find_marker_func``. A geometric ROI
does not establish grid order, physical marker identity, or tracking validity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def load_marker_roi(path: str | Path) -> dict:
    """Load a versioned ROI; validation is shared with the detector factory."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_marker_roi(config)
    return config


def validate_marker_roi(config: dict) -> None:
    if config.get("schema") != "revo3-marker-roi-v1":
        raise ValueError("Unsupported marker ROI schema")
    if not isinstance(config.get("camera_serial"), str) or not config["camera_serial"]:
        raise ValueError("A camera serial is required")
    if config.get("coordinate_space") != "raw_rgb_pixels":
        raise ValueError("This adapter expects raw, unresized image coordinates")
    hw = config.get("image_hw")
    if (not isinstance(hw, list) or len(hw) != 2
            or any(type(n) is not int or n <= 0 for n in hw)):
        raise ValueError("image_hw must contain positive integer height and width")
    if not isinstance(config.get("include_polygons"), list) or not config["include_polygons"]:
        raise ValueError("At least one closed inclusion polygon is required")
    if not isinstance(config.get("exclude_polygons", []), list):
        raise ValueError("exclude_polygons must be a list")
    for polygon in config["include_polygons"] + config.get("exclude_polygons", []):
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Each polygon needs at least three [x, y] vertices")
        if not np.isfinite(points).all():
            raise ValueError("Polygon vertices must be finite")
        if (points[:, 0].min() < 0 or points[:, 0].max() > hw[1] - 1
                or points[:, 1].min() < 0 or points[:, 1].max() > hw[0] - 1):
            raise ValueError("Polygon vertices fall outside the calibrated image")
        if abs(cv2.contourArea(points)) < 1:
            raise ValueError("Polygon area is too small")


def marker_centers_in_roi(centers: np.ndarray, config: dict) -> np.ndarray:
    """Classify centers geometrically; inclusion edges are kept, holes excluded."""
    validate_marker_roi(config)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    include = [np.asarray(p, dtype=np.float32) for p in config["include_polygons"]]
    exclude = [np.asarray(p, dtype=np.float32) for p in config.get("exclude_polygons", [])]
    accepted = []
    for center in centers:
        point = tuple(map(float, center))
        inside = np.isfinite(center).all() and any(
            cv2.pointPolygonTest(p, point, False) >= 0 for p in include
        )
        in_hole = inside and any(
            cv2.pointPolygonTest(p, point, False) >= 0 for p in exclude
        )
        accepted.append(inside and not in_hole)
    return np.asarray(accepted, dtype=bool)


def filter_marker_components(mask: np.ndarray, config: dict) -> np.ndarray:
    """Keep entire detected components whose centroids lie inside the ROI.

    Applying the ROI to the binary detector's components avoids introducing
    dark RGB edges and avoids cutting a real marker into a displaced centroid.
    The mask is a center-selection region, not a per-pixel clipping region.
    """
    validate_marker_roi(config)
    mask = np.asarray(mask)
    if mask.ndim != 2 or list(mask.shape) != config["image_hw"]:
        raise ValueError("Detector mask size differs from the calibrated ROI")
    if not np.isfinite(mask).all():
        raise ValueError("Detector mask must be finite")
    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    _, labels, _, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    accepted = marker_centers_in_roi(centers[1:], config)
    kept_labels = np.flatnonzero(accepted) + 1
    return np.where(np.isin(labels, kept_labels), 255, 0).astype(np.uint8)


def make_roi_detector(
    config: dict, camera_serial: str,
    detector: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callback for raw BGR frames; reject serial or resolution mismatch.

    SDK loading is deferred. For a rectified/warped image, transform the ROI
    into that image's coordinates and use a corresponding adapter instead.
    """
    validate_marker_roi(config)
    if camera_serial != config["camera_serial"]:
        raise ValueError("ROI belongs to another camera")
    # Freeze the configuration so a GUI edit cannot silently change an active tracker.
    config = json.loads(json.dumps(config))
    if detector is None:
        from vitai_marker.dfs_tracking import find_marker
        detector = find_marker

    def find_marker_with_roi(frame_bgr: np.ndarray) -> np.ndarray:
        if (frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3
                or list(frame_bgr.shape[:2]) != config["image_hw"]):
            raise ValueError("Expected uint8 BGR in the calibrated raw image size")
        return filter_marker_components(detector(frame_bgr), config)

    return find_marker_with_roi
