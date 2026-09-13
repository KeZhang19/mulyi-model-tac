"""Move existing rubber-frame calibration into Direct Revo3 DIP frames."""

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


CALIBRATION_FINGERS = ("middle", "index", "ring", "pinky", "thumb")
DIRECT_LINKS = (
    "right_middle_DIP_Link", "right_index_DIP_Link", "right_ring_DIP_Link",
    "right_little_DIP_Link", "right_thumb_DIP_Link",
)
RUBBER_STEMS = ("mid", "index", "ring", "pinky", "thumb")


def rubber_to_dip_transform(urdf: str | Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    joint = ET.parse(urdf).find(f".//joint[@name='right_{stem}dip_roll_rubber_joint']")
    if joint is None or joint.attrib.get("type") != "fixed":
        raise ValueError(f"Missing fixed rubber-to-DIP joint for {stem}")
    origin = joint.find("origin")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    roll, pitch, yaw = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                         (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr)))
    return xyz, rotation


def prepare_direct_calibration(layout_path: str | Path, urdf: str | Path, output_dir: str | Path) -> Path:
    """Transform positions/directions, retaining calibrated camera pixels and IDs.

    The source assets remain untouched. The middle rubber's fixed joint carries
    a roughly 29.5 mm translation, so simply changing link names is incorrect.
    """
    layout_path, output_dir = Path(layout_path), Path(output_dir)
    with np.load(layout_path, allow_pickle=False) as source:
        layout = {key: source[key].copy() for key in source.files}
    for finger, stem in zip(CALIBRATION_FINGERS, RUBBER_STEMS, strict=True):
        position, rotation = rubber_to_dip_transform(urdf, stem)
        for key, value in tuple(layout.items()):
            if not key.startswith(finger + "_"):
                continue
            if key.endswith(("_points_link_m", "_starts_link_m", "_origin_link_m")):
                layout[key] = (value @ rotation.T + position).astype(np.float32)
            elif key.endswith(("_normals_link", "_directions_link")):
                layout[key] = (value @ rotation.T).astype(np.float32)
            elif key.endswith("_camera_rotation_link"):
                layout[key] = (rotation @ value).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "marker_positions.npz"
    np.savez(output, **layout)
    rectangle_name = "camera_ray_rectangles_320x240.json"
    shutil.copyfile(layout_path.with_name(rectangle_name), output_dir / rectangle_name)
    return output
