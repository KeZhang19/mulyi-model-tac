from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_utlis_misc_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "official_replay" / "utils" / "utlis_misc.py"
    )
    spec = importlib.util.spec_from_file_location("utlis_misc", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def test_revo21_mid_links_are_middle_finger_slot():
    module = _load_utlis_misc_module()

    assert module._infer_finger("/World/Robot/right_midpip_roll_touch_link") == "middle"
    assert module._sensor_slot("/World/Robot/right_midpip_roll_touch_link") == 2
    assert module._sensor_slot("/World/Robot/right_indexpip_roll_touch_link") == 1
