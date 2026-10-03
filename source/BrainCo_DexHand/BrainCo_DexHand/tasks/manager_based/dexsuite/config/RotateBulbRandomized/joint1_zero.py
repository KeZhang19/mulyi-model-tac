"""Task-owned joint1 coordinate origin: the original reset pose is zero."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
JOINT1_ZERO_USD = ROOT / "assets/rotate_bulb_randomized/rizon4_joint1_zero.usda"
JOINT1_ZERO_WORKSPACE = ROOT / "assets/rotate_bulb_randomized/workspace_bank_joint1_zero_v1.json"
# q_original = q_zero + this angle. Only joint1 changes coordinates.
JOINT1_ZERO_OFFSET_RAD = -0.14578291835403787
