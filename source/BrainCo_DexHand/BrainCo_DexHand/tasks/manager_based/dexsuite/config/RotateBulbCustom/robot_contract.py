"""Task-owned Flexiv Rizon4 / Revo3 asset names and initial pose."""

import json
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parents[8] / "assets/rotate_bulb_custom"
SOURCE_TASK = "BrainCo-Dexsuite-Flexiv-Lift-Distillation-v2-test"
GRASP_REFERENCE_PATH = ASSET_ROOT / "grasp_reference/simulation_state.json"
GRASP_REFERENCE = json.loads(GRASP_REFERENCE_PATH.read_text(encoding="utf-8"))
if GRASP_REFERENCE.get("schema") != "revo3_simulation_state" or GRASP_REFERENCE.get("schema_version") != 1:
    raise ValueError(f"Unsupported grasp reference: {GRASP_REFERENCE_PATH}")
ROBOT_POSITION = tuple(GRASP_REFERENCE["robot"]["base"]["position_m"])
ROBOT_ORIENTATION = tuple(GRASP_REFERENCE["robot"]["base"]["quaternion_wxyz"])

# Asset joint inventory, ordered by kinematic depth. This does not define an
# action transform: Custom keeps its existing relative joint position control.
JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "right_index_MPR_joint", "right_little_MPR_joint", "right_middle_MPR_joint", "right_ring_MPR_joint",
    "right_thumb_CMP_joint",
    "right_index_MCP_joint", "right_little_MCP_joint", "right_middle_MCP_joint", "right_ring_MCP_joint",
    "right_thumb_CMR_joint",
    "right_index_PIP_joint", "right_little_PIP_joint", "right_middle_PIP_joint", "right_ring_PIP_joint",
    "right_thumb_MCP_joint",
    "right_index_DIP_joint", "right_little_DIP_joint", "right_middle_DIP_joint", "right_ring_DIP_joint",
    "right_thumb_PIP_joint", "right_thumb_DIP_joint",
)
# The export groups fingers; PhysX orders joints by depth. Always map by name.
INITIAL_JOINT_POS = dict(zip(
    GRASP_REFERENCE["robot"]["joint_names"],
    GRASP_REFERENCE["robot"]["joint_positions_rad"],
    strict=True,
))
if set(INITIAL_JOINT_POS) != set(JOINT_NAMES):
    raise ValueError("Grasp reference must contain exactly the 28 Flexiv/Revo3 joints")
HAND_POINT_NAMES = (
    "palm", "right_index_tip_Link", "right_little_tip_Link",
    "right_middle_tip_Link", "right_ring_tip_Link", "right_thumb_tip_Link",
)

# Preserve the tactile encoder's collection order, independently of joint order.
FINGER_ORDER = ("middle", "index", "ring", "pinky", "thumb")
FINGER_LINKS = {
    "middle": "right_middle_DIP_Link", "index": "right_index_DIP_Link",
    "ring": "right_ring_DIP_Link", "pinky": "right_little_DIP_Link", "thumb": "right_thumb_DIP_Link",
}
TACTILE_LINK_MAP = {
    "right_middip_roll_rubber_link": FINGER_LINKS["middle"],
    "right_indexdip_roll_rubber_link": FINGER_LINKS["index"],
    "right_ringdip_roll_rubber_link": FINGER_LINKS["ring"],
    "right_pinkydip_roll_rubber_link": FINGER_LINKS["pinky"],
    "right_thumbdip_roll_rubber_link": FINGER_LINKS["thumb"],
    "right_hand_rubber_link": "palm",
}
for _old, _new in (("mid", "middle"), ("index", "index"), ("ring", "ring"), ("pinky", "little"), ("thumb", "thumb")):
    for _segment in ("mcp", "pip"):
        TACTILE_LINK_MAP[f"right_{_old}{_segment}_roll_touch_link"] = f"right_{_new}_{_segment.upper()}_Link"
