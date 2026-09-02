import os

import xml.etree.ElementTree as ET

def _parse_elastomer_origins(urdf_path: str) -> dict:
    """Return tactile fixed-joint origins found in the robot URDF."""
    urdf_path = os.path.expanduser(str(urdf_path))
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)

    out = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        name = joint.get("name", "").lower()
        child = joint.find("child")
        child_link = child.get("link", "").lower() if child is not None else ""

        if "elastomer_joint_left" in name:
            key = "left"
        elif "elastomer_joint_right" in name:
            key = "right"
        elif "touch_link" in child_link:
            key = child_link
        else:
            continue

        origin = joint.find("origin")
        if origin is None:
            continue
        try:
            xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
            if len(xyz) != 3 or len(rpy) != 3:
                continue
        except (ValueError, TypeError):
            continue

        out[key] = (xyz, rpy)

    return out


def _infer_arm(link_path: str) -> str | None:
    lp = link_path.lower()
    if "left_arm_" in lp or "/left/" in lp or "/left_" in lp or "left_" in lp:
        return "left"
    if "right_arm_" in lp or "/right/" in lp or "/right_" in lp or "right_" in lp:
        return "right"
    return None


def _infer_finger(link_path: str) -> str | None:
    lp = link_path.lower()
    if "elastomer_left" in lp or "_left_finger_link" in lp:
        return "left_finger"
    if "elastomer_right" in lp or "_right_finger_link" in lp:
        return "right_finger"
    aliases = {
        "thumb": ("thumb",),
        "index": ("index",),
        "middle": ("middle", "mid"),
        "ring": ("ring",),
        "pinky": ("pinky",),
    }
    for finger, names in aliases.items():
        if any(f"_{name}" in lp or f"/{name}" in lp for name in names):
            return finger
    return None


def _sensor_slot(link_path: str) -> int | None:
    """Canonical slot: 0=L/L, 1=L/R, 2=R/L, 3=R/R."""
    lp = link_path.lower()
    revo2_order = {
        "thumb": 0,
        "index": 1,
        "middle": 2,
        "ring": 3,
        "pinky": 4,
    }
    aliases = {
        "thumb": ("thumb",),
        "index": ("index",),
        "middle": ("middle", "mid"),
        "ring": ("ring",),
        "pinky": ("pinky",),
    }
    for finger, slot in revo2_order.items():
        if any(f"_{name}" in lp or f"/{name}" in lp for name in aliases[finger]):
            return slot

    arm, finger = _infer_arm(link_path), _infer_finger(link_path)
    if arm is None or finger is None:
        return None
    return (0 if arm == "left" else 2) + (0 if finger == "left_finger" else 1)
