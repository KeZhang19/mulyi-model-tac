"""Fixed marker IDs in raw pixels, with conservative, stateless association.

This is a small-displacement adapter around SDK detections, not DFS tracking.
Disjoint reference gates reject ambiguous detections; no missing point is filled.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

try:
    from .marker_roi import marker_centers_in_roi, validate_marker_roi
    from .marker_training import select_marker_ids
except ImportError:  # Direct script entry points.
    from marker_roi import marker_centers_in_roi, validate_marker_roi
    from marker_training import select_marker_ids


def roi_identity(config):
    """Ignore selection timestamps, but bind IDs to the exact camera and region."""
    keys = ("camera_serial", "coordinate_space", "image_hw", "include_polygons")
    geometry = {key: config[key] for key in keys}
    geometry["exclude_polygons"] = config.get("exclude_polygons", [])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def create_marker_template(points, config, max_displacement_px=12.0):
    validate_marker_roi(config)
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 1000
            or not np.isfinite(points).all()):
        raise ValueError("Reference needs 1..1000 finite [x, y] points")
    if not marker_centers_in_roi(points, config).all():
        raise ValueError("All reference points must be inside the ROI")
    if not np.isfinite(max_displacement_px) or not 0 < max_displacement_px <= 50:
        raise ValueError("max_displacement_px must be in (0, 50]")
    # Sort only at template creation. IDs never depend on later detector order.
    points = points[np.lexsort((points[:, 0], points[:, 1]))]
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    spacing = distance.min(axis=1)
    if (spacing < 2).any():
        raise ValueError("Reference points must be at least 2 pixels apart")
    gates = np.minimum(max_displacement_px, .4 * spacing)
    # Uniform coverage for N > 100, chosen once; retain every reference separately.
    output_ids = select_marker_ids(points)
    return {
        "schema": "revo3-marker-template-v1", "camera_serial": config["camera_serial"],
        "coordinate_space": "raw_rgb_pixels", "image_hw": config["image_hw"],
        "roi_sha256": roi_identity(config), "reference_xy": points.tolist(),
        "reference_ids": list(range(len(points))), "output_ids": output_ids,
        "gate_radius_px": gates.tolist(), "max_displacement_px": float(max_displacement_px),
        "association": "unique_candidate_in_disjoint_reference_gate",
        "features": ["x0", "y0", "dx", "dy", "valid"],
        "invalid_features": "all_zero; origins retained in reference_xy",
        "contact_state": "unverified",
    }


class FixedMarkerMatcher:
    def __init__(self, template, config):
        # Rebuild the deterministic contract to reject malformed or altered IDs/gates.
        expected = create_marker_template(template["reference_xy"], config,
                                          template["max_displacement_px"])
        for key in expected:
            if key != "contact_state" and template.get(key) != expected[key]:
                raise ValueError(f"Template mismatch: {key}")
        self.template = json.loads(json.dumps(template))
        self.config = json.loads(json.dumps(config))
        self.reference = np.asarray(template["reference_xy"], dtype=np.float64)
        self.gates = np.asarray(template["gate_radius_px"], dtype=np.float64)
        self.output_ids = np.asarray(template["output_ids"], dtype=np.int64)

    def match(self, centers, camera_serial, image_hw):
        if camera_serial != self.template["camera_serial"] or list(image_hw) != self.template["image_hw"]:
            raise ValueError("Frame camera or image size differs from the reference")
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
        inside = marker_centers_in_roi(centers, self.config)
        distance = np.linalg.norm(self.reference[:, None] - centers[None, :], axis=2)
        eligible = (distance <= self.gates[:, None]) & inside[None, :]
        # Zero or multiple candidates in a gate -> missing. Never resolve a tie by order.
        valid = eligible.sum(axis=1) == 1
        indices = np.full(len(self.reference), -1, dtype=np.int64)
        if len(centers):
            indices[valid] = np.argmax(eligible[valid], axis=1)
        positions = np.zeros_like(self.reference)
        positions[valid] = centers[indices[valid]]
        marker = np.zeros((100, 5), dtype=np.float32)
        rows = np.flatnonzero(self.output_ids >= 0)
        rows = rows[valid[self.output_ids[rows]]]
        ids = self.output_ids[rows]
        marker[rows, :2] = self.reference[ids]
        marker[rows, 2:4] = positions[ids] - self.reference[ids]
        marker[rows, 4] = 1
        matched_candidates = indices[valid]
        return {
            "marker": marker, "reference_valid": valid,
            "matched_candidate_indices": indices, "current_xy": positions,
            "candidate_in_roi": inside,
            "matched": int(valid.sum()), "missing": int((~valid).sum()),
            "extra": int(inside.sum() - len(matched_candidates)),
            "ambiguous": int((eligible.sum(axis=1) > 1).sum()),
            "model_valid": int(marker[:, 4].sum()),
            "padding": int((self.output_ids < 0).sum()),
        }


def save_marker_batch(path, matcher, samples):
    """Store fixed rows and all variable detections without pickle/object arrays."""
    results = [matcher.match(s["centers"], matcher.template["camera_serial"],
                             s["bgr"].shape[:2]) for s in samples]
    counts = np.array([len(s["centers"]) for s in samples], dtype=np.int64)
    np.savez_compressed(
        path, marker=np.stack([r["marker"] for r in results]),
        reference_xy=matcher.reference, output_ids=matcher.output_ids,
        reference_valid=np.stack([r["reference_valid"] for r in results]),
        matched_candidate_indices=np.stack([r["matched_candidate_indices"] for r in results]),
        current_xy=np.stack([r["current_xy"] for r in results]),
        candidate_xy=np.concatenate([np.asarray(s["centers"]).reshape(-1, 2) for s in samples]),
        candidate_offsets=np.r_[0, np.cumsum(counts)],
        candidate_in_roi=np.concatenate([r["candidate_in_roi"] for r in results]),
        sequence=np.array([s["sequence"] for s in samples], dtype=np.int64),
        host_read_end_unix_ns=np.array([s["host_read_end_unix_ns"] for s in samples], dtype=np.int64),
    )
    return {"shape": [len(samples), 100, 5], "reference_points": len(matcher.reference),
            "padding_rows": int((matcher.output_ids < 0).sum()),
            "valid_range": [min(r["model_valid"] for r in results), max(r["model_valid"] for r in results)],
            "ambiguous_point_frames": sum(r["ambiguous"] for r in results),
            "coordinate_space": "raw_rgb_pixels", "association": matcher.template["association"]}
