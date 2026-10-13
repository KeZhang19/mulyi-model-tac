"""Fixed real-dot identities for 100-row training inputs; never interpolate flow."""
from __future__ import annotations

import numpy as np


def select_marker_ids(reference, count=100):
    """Select coverage once on the reference; -1 denotes an invalid padding row."""
    points = np.asarray(reference, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points)):
        raise ValueError("Expected distinct finite reference dots [N,2]")
    selected = list(range(len(points)))
    if len(points) > count:
        selected = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
        nearest = np.linalg.norm(points - points[selected[0]], axis=1)
        while len(selected) < count:
            index = int(np.argmax(nearest))
            selected.append(index)
            nearest = np.minimum(nearest, np.linalg.norm(points - points[index], axis=1))
        selected.sort()
    return selected + [-1] * (count - len(selected))


def create_dfs_layout(origin, image_hw, camera_serial, reference_sha256):
    origin = np.asarray(origin)
    if origin.ndim not in (2, 3) or origin.shape[-1] != 2:
        raise ValueError("DFS origin must have shape [N,2] or [rows,columns,2]")
    h, w = map(int, image_hw)
    points = origin.reshape(-1, 2)
    ids = select_marker_ids(points)
    if (min(h, w) < 2 or (points < 0).any() or (points > [w - 1, h - 1]).any()
            or not camera_serial or not reference_sha256):
        raise ValueError("DFS reference identity or image coordinates invalid")
    return {"schema": "dfs-fixed-marker-layout-v1", "backend": "sdk_marker_tracking",
            "coordinate_space": "vitai_warped_img_pixels", "image_hw": [h, w],
            "camera_serial": camera_serial, "reference_sha256": reference_sha256,
            "native_shape": list(origin.shape), "reference_xy": points.tolist(),
            "output_ids": ids, "features": ["u0", "v0", "du", "dv", "valid"],
            "selection": "reference_farthest_point_coverage; fixed_native_ids; no_interpolation",
            "invalid_features": "all_zero; original coordinates retained in source bundles"}


def pack_dfs_frame(origin, current, matched, layout):
    """Keep missing IDs in place; no new detections or interpolation fill holes."""
    origin, current, matched = map(np.asarray, (origin, current, matched))
    expected = create_dfs_layout(origin, layout["image_hw"], layout["camera_serial"],
                                 layout["reference_sha256"])
    if expected != layout:
        raise ValueError("DFS reference/layout changed; start a separate source session")
    if current.shape != origin.shape or matched.shape != origin.shape[:-1] or matched.dtype != bool:
        raise ValueError("Recorded DFS current/matched shapes or mask dtype invalid")
    points, positions, mask = origin.reshape(-1, 2), current.reshape(-1, 2), matched.reshape(-1)
    h, w = layout["image_hw"]
    if (not np.isfinite(positions[mask]).all() or (positions[mask] < 0).any()
            or (positions[mask] > [w - 1, h - 1]).any()):
        raise ValueError("Matched DFS position is nonfinite or outside the image")
    output_ids = np.asarray(layout["output_ids"], dtype=np.int64)
    rows = np.flatnonzero(output_ids >= 0)
    rows = rows[mask[output_ids[rows]]]
    ids = output_ids[rows]
    marker = np.zeros((100, 5), np.float32)
    marker[rows, :2] = points[ids]
    marker[rows, 2:4] = positions[ids] - points[ids]
    marker[rows, 4] = 1
    return marker, marker[:, 4].astype(bool)
