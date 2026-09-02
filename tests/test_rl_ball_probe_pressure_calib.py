from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "BrainCo_DexHand"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from BrainCo_DexHand.force_map.pressure_trace_metrics import validate_pressure_trace_v1
from BrainCo_DexHand.force_map.rl_ball_probe_pressure_calib import (
    BallProbePressureCalibState,
    RlPressureTraceRecorder,
    validate_press_indent_depth,
)


def test_press_indent_depth_has_0p5mm_hard_limit():
    validate_press_indent_depth(0.0005)

    with pytest.raises(ValueError, match="0.5mm"):
        validate_press_indent_depth(0.0005001)


def test_ball_probe_calib_state_searches_contact_then_steps_indent():
    state = BallProbePressureCalibState(
        search_distance_m=0.02,
        indent_depth_m=0.0005,
        indent_steps=3,
        settle_steps=2,
        contact_penetration_threshold_m=1.0e-7,
        contact_force_threshold_n=0.01,
    )

    assert state.desired_travel_m == pytest.approx(0.02)
    assert not state.observe(step=0, penetration_max_m=0.0, normal_force_n=0.0, actual_axis_disp_m=0.001)

    assert not state.observe(step=1, penetration_max_m=2.0e-7, normal_force_n=0.0, actual_axis_disp_m=0.004)
    assert state.contact_found
    assert state.contact_step == 1
    assert state.desired_travel_m == pytest.approx(0.004)
    assert state.commanded_indent_m == pytest.approx(0.0)

    assert not state.observe(step=2, penetration_max_m=2.0e-7, normal_force_n=0.02, actual_axis_disp_m=0.004)
    assert state.observe(step=3, penetration_max_m=2.0e-7, normal_force_n=0.02, actual_axis_disp_m=0.004)
    state.mark_sample_recorded()

    assert state.commanded_indent_m == pytest.approx(0.00025)
    assert state.desired_travel_m == pytest.approx(0.00425)


def test_rl_pressure_trace_recorder_writes_v1_trace(tmp_path):
    sensors, rows, cols = 1, 1, 39
    layout = {
        "pressure_taxel_points_l_m": np.zeros((sensors, rows, cols, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((sensors, rows, cols, 3), dtype=np.float32),
        "pressure_taxel_layout_valid": np.ones((sensors,), dtype=np.uint8),
    }
    recorder = RlPressureTraceRecorder(
        tmp_path,
        {
            "run_id": "unit",
            "pressure_backend_id": "rl_warpsdf_dynamic_axis",
            "pressure_layout_id": "unit:index_mcp:1x39",
            "pressure_calibration_id": "rl_inline",
        },
        layout_arrays=layout,
    )

    base = np.zeros((sensors, rows, cols), dtype=np.float32)
    recorder.record(
        step=0,
        phase="indent",
        pressure_norm=base,
        pressure_raw_n=base,
        penetration_m=base,
        signed_distance_m=base,
        commanded_indent_m=0.0,
        actual_axis_disp_m=0.004,
        actual_post_contact_indent_m=0.0,
        contact_force_n=0.02,
        contact_normal_force_n=0.02,
        contact_found=True,
    )
    recorder.record(
        step=1,
        phase="indent",
        pressure_norm=base + 0.1,
        pressure_raw_n=base + 0.2,
        penetration_m=base + 1.0e-5,
        signed_distance_m=base - 1.0e-5,
        commanded_indent_m=0.00002,
        actual_axis_disp_m=0.00402,
        actual_post_contact_indent_m=0.00002,
        contact_force_n=0.03,
        contact_normal_force_n=0.03,
        contact_found=True,
    )

    npz_path, metadata_path = recorder.close()

    assert metadata_path.exists()
    result = validate_pressure_trace_v1(npz_path)
    assert result["passed"], result
    assert {key: result[key] for key in ("num_steps", "sensor_count", "image_shape")} == {
        "num_steps": 2,
        "sensor_count": 1,
        "image_shape": [1, 39],
    }
