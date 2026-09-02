from __future__ import annotations

import ast
import json
import importlib.util
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from BrainCo_DexHand.force_map import (
    AnalyticPressTrajectory,
    AnalyticPresserSpec,
    CalibratedPressureMapSensor,
    GeometryNormalRayPenetrationSource,
    PenetrationFrame,
    NormalRayPenetrationSource,
    PhysxContactSource,
    PressureCalibration,
    PressureAreaFractionFitConfig,
    PressureDeadbandFitConfig,
    PressureContactEventBatch,
    PressureFrame,
    PressureSampleSupportFitConfig,
    PressureSpatialFootprintFitConfig,
    PressureTaxelMap,
    TaxelLayout,
    UrdfPressureLayoutSource,
    UrdfPressurePadSpec,
    WarpSdfPenetrationSource,
    align_reference_to_pressure_taxels,
    aligned_reference_points_for_pressure_taxels,
    apply_pressure_area_fraction_to_trace,
    apply_geometry_normal_ray_to_trace,
    apply_normal_ray_reference_to_trace,
    apply_pressure_deadband_to_trace,
    apply_pressure_mask_depth_to_trace,
    apply_pressure_sample_support_to_trace,
    apply_pressure_spatial_footprint_to_trace,
    apply_reference_alignment_to_values,
    build_reference_to_pressure_alignment,
    fit_pressure_area_fraction_to_reference,
    fit_pressure_deadband_to_reference,
    fit_pressure_mask_depth_to_reference,
    fit_pressure_sample_support_to_reference,
    fit_pressure_spatial_footprint_to_reference,
    UrdfPressureContactAdapter,
    calibrate_gap_fraction,
    calibrate_penetration,
    analytic_presser_penetration,
    evaluate_pressure_trace_report,
    generate_l0_pressure_trace,
    infer_pressure_reference_layer,
    load_pressure_calibration_overrides,
    load_pressure_pad_specs_from_urdf,
    load_pressure_taxel_maps_from_urdf,
    load_pressure_touch_links_from_urdf,
    load_touch_links_from_urdf,
    pressure_map_stats,
    pressure_reference_frame_diagnostics,
    pressure_reference_origin_alignment_diagnostics,
    pressure_sample_boundary_diagnostics,
    pressure_trace_report,
    select_pressure_pad_spec,
    triangle_mesh_topology_diagnostics,
    validate_pressure_trace_v1,
    weld_duplicate_triangle_vertices,
    write_geometry_aligned_trace,
    write_geometry_normal_ray_trace,
    write_normal_ray_reference_trace,
)


def _grid(calibration: PressureCalibration | None = None) -> PressureTaxelMap:
    return PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=5,
        num_cols=5,
        point_distance=1.0,
        normal_axis=2,
        normal_offset=0.0,
        calibration=calibration or PressureCalibration(stiffness=1.0, max_force=10.0),
    )


def test_pressure_taxel_grid_accepts_rectangular_pitch():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_pad",
        num_rows=2,
        num_cols=3,
        row_distance=0.002,
        col_distance=0.006,
        normal_axis=2,
    )

    points = np.asarray(taxel_map.points_l).reshape(2, 3, 3)

    np.testing.assert_allclose(points[:, 0, 0], [-0.001, 0.001], atol=1.0e-9)
    np.testing.assert_allclose(points[0, :, 1], [-0.006, 0.0, 0.006], atol=1.0e-9)
    np.testing.assert_allclose(points[..., 2], 0.0, atol=1.0e-9)


def test_penetration_calibration_matches_penalty_spring_defaults():
    penetration = np.array([0.0, 0.001, 0.003], dtype=np.float32)
    raw, normalized = calibrate_penetration(
        penetration,
        PressureCalibration(stiffness=5_000.0, max_force=10.0),
    )

    np.testing.assert_allclose(raw, np.array([0.0, 5.0, 10.0], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(normalized, np.array([0.0, 0.5, 1.0], dtype=np.float32), atol=1.0e-6)


def test_zero_penetration_stays_zero_even_with_bias():
    penetration = np.zeros((3,), dtype=np.float32)
    raw, normalized = calibrate_penetration(
        penetration,
        PressureCalibration(stiffness=5_000.0, max_force=10.0, bias=1.0),
    )

    np.testing.assert_allclose(raw, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(normalized, 0.0, atol=1.0e-6)


def test_kelvin_voigt_damping_adds_force_only_during_contact():
    penetration = np.array([0.0, 0.001, 0.001], dtype=np.float32)
    velocity = np.array([1.0, 0.0, 0.2], dtype=np.float32)

    raw, normalized = calibrate_penetration(
        penetration,
        PressureCalibration(stiffness=1_000.0, damping=10.0, max_force=10.0),
        penetration_velocity=velocity,
    )

    np.testing.assert_allclose(raw, np.array([0.0, 1.0, 3.0], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(normalized, np.array([0.0, 0.1, 0.3], dtype=np.float32), atol=1.0e-6)


def test_gap_fraction_calibration_ignores_spring_and_damping():
    closure = np.array([0.0, 0.00025, 0.0005, 0.002], dtype=np.float32)
    raw, normalized = calibrate_gap_fraction(
        closure,
        0.001,
        PressureCalibration(stiffness=99_999.0, damping=99_999.0, max_force=8.0),
    )

    np.testing.assert_allclose(raw, np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32), atol=1.0e-6)
    np.testing.assert_allclose(normalized, np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32), atol=1.0e-6)


def test_empty_contacts_stay_zero_even_with_bias():
    taxel_map = _grid(PressureCalibration(stiffness=1.0, max_force=10.0, bias=1.0))
    events = PressureContactEventBatch(
        source_link="finger_tip",
        contact_points_l=np.zeros((0, 3), dtype=np.float32),
        contact_normals_l=np.zeros((0, 3), dtype=np.float32),
        normal_forces=np.zeros((0,), dtype=np.float32),
    )

    output = taxel_map.force_from_contacts(events)

    np.testing.assert_allclose(output.raw_force_map, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(output.force_map, 0.0, atol=1.0e-6)


def test_single_contact_spreads_to_center_taxel():
    taxel_map = _grid()
    events = PressureContactEventBatch(
        source_link="finger_tip",
        contact_points_l=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        contact_normals_l=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        normal_forces=np.array([5.0], dtype=np.float32),
    )

    output = taxel_map.force_from_contacts(events, kernel_sigma=0.6, conserve_total_force=False)
    force_map = output.force_map

    assert force_map[2, 2] == np.max(force_map)
    assert force_map[2, 2] > force_map[0, 0]


def test_multiple_contacts_accumulate_without_overwriting():
    taxel_map = _grid()
    events = PressureContactEventBatch(
        source_link="finger_tip",
        contact_points_l=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        contact_normals_l=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        normal_forces=np.array([2.0, 3.0], dtype=np.float32),
    )

    output = taxel_map.force_from_contacts(events, kernel_sigma=0.25, conserve_total_force=False)
    force_map = output.force_map

    assert force_map[1, 2] > 0.0
    assert force_map[3, 2] > 0.0
    assert force_map[1, 2] < force_map[3, 2]


def test_opposite_normal_is_filtered_out():
    taxel_map = _grid()
    events = PressureContactEventBatch(
        source_link="finger_tip",
        contact_points_l=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        contact_normals_l=np.array([[0.0, 0.0, -1.0]], dtype=np.float32),
        normal_forces=np.array([5.0], dtype=np.float32),
    )

    output = taxel_map.force_from_contacts(events, kernel_sigma=0.6)

    assert np.count_nonzero(output.force_map) == 0


def test_calibrated_pressure_map_sensor_accepts_stacked_penetrations():
    taxel_map = _grid(PressureCalibration(stiffness=2.0, max_force=4.0))
    sensor = CalibratedPressureMapSensor([taxel_map])
    penetration = np.zeros((1, 5, 5), dtype=np.float32)
    penetration[0, 2, 2] = 1.0

    output = sensor.from_penetrations(penetration)

    assert output.force_map.shape == (1, 5, 5)
    assert output.force_map[0, 2, 2] == 0.5


def test_pressure_frame_reports_total_force_and_center_of_pressure():
    raw = np.zeros((1, 1, 5, 5), dtype=np.float32)
    raw[0, 0, 1, 3] = 2.0
    raw[0, 0, 3, 1] = 2.0
    norm = raw / 4.0

    total, center = pressure_map_stats(raw)
    frame = PressureFrame.from_maps(raw, norm, calibration_id="unit")

    np.testing.assert_allclose(total, np.array([[4.0]], dtype=np.float32))
    np.testing.assert_allclose(center, np.array([[[2.0, 2.0]]], dtype=np.float32))
    np.testing.assert_allclose(frame.total_force_n, total)
    np.testing.assert_allclose(frame.center_of_pressure_px, center)
    assert frame.calibration_id == "unit"


def test_taxel_layout_and_penetration_frame_hold_canonical_shapes():
    taxel_map = _grid()
    layout = TaxelLayout.from_taxel_map(taxel_map, sensor_id=3, source="test")
    penetration = np.zeros((1, 1, 5, 5), dtype=np.float32)
    frame = PenetrationFrame(penetration_m=penetration, contact_mask=penetration > 0.0)

    assert layout.sensor_id == 3
    assert layout.link_name == "finger_tip"
    assert layout.image_shape == (5, 5)
    assert layout.source == "test"
    assert frame.penetration_m.shape == (1, 1, 5, 5)
    assert frame.contact_mask.shape == (1, 1, 5, 5)


def test_urdf_pressure_contact_adapter_feeds_same_pipeline():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])
    events_by_link = adapter.events_by_link(
        [
            {
                "source_link": "finger_tip",
                "contact_point_l": [0.0, 0.0, 0.0],
                "contact_normal_l": [0.0, 0.0, 1.0],
                "normal_force": 4.0,
            }
        ]
    )

    output = CalibratedPressureMapSensor([taxel_map], kernel_sigma=0.5).from_contact_events(events_by_link)

    assert output.force_map.shape == (1, 5, 5)
    assert output.force_map[0, 2, 2] > 0.0


def test_urdf_pressure_contact_adapter_accepts_contact_aliases():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])

    events = adapter.events_by_link(
        [
            {
                "body_name": "finger_tip",
                "point_l": [0.0, 0.0, 0.0],
                "normal_l": [0.0, 0.0, 2.0],
                "force_n": 3.0,
                "tangent_force_l": [0.1, 0.0, 0.0],
            }
        ]
    )

    output = CalibratedPressureMapSensor([taxel_map], kernel_sigma=0.5).from_contact_events(events)

    assert float(events["finger_tip"].normal_forces[0]) == 3.0
    np.testing.assert_allclose(events["finger_tip"].contact_normals_l[0], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(events["finger_tip"].shear_forces_l[0], np.array([0.1, 0.0, 0.0]))
    assert output.force_map[0, 2, 2] > 0.0


def test_urdf_pressure_contact_adapter_projects_force_vector_to_normal_force():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])

    events = adapter.events_by_link(
        [
            {
                "source_link": "finger_tip",
                "contact_point_l": [0.0, 0.0, 0.0],
                "contact_normal_l": [0.0, 0.0, 2.0],
                "contact_force_l": [1.0, 0.0, 4.0],
            }
        ]
    )

    output = CalibratedPressureMapSensor([taxel_map], kernel_sigma=0.5).from_contact_events(events)

    assert float(events["finger_tip"].normal_forces[0]) == 4.0
    assert output.force_map[0, 2, 2] > 0.0


def test_urdf_pressure_contact_adapter_treats_vector_force_as_vector():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])

    events = adapter.events_by_link(
        [
            {
                "source_link": "finger_tip",
                "contact_point_l": [0.0, 0.0, 0.0],
                "contact_normal_l": [0.0, 0.0, 1.0],
                "force": [1.0, 0.0, 4.0],
            }
        ]
    )

    assert float(events["finger_tip"].normal_forces[0]) == 4.0


def test_urdf_pressure_contact_adapter_treats_vector_force_with_world_normal_as_world():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])
    quat_x90 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)

    events = adapter.events_by_link(
        [
            {
                "source_link": "finger_tip",
                "contact_point_l": [0.0, 0.0, 0.0],
                "contact_normal_w": [0.0, 0.0, 1.0],
                "force": [0.0, 0.0, 4.0],
            }
        ],
        link_poses_w={"finger_tip": (np.zeros(3, dtype=np.float32), quat_x90)},
    )

    assert float(events["finger_tip"].normal_forces[0]) == 4.0


def test_urdf_pressure_contact_adapter_projects_world_force_with_local_normal():
    taxel_map = _grid()
    adapter = UrdfPressureContactAdapter([taxel_map])

    events = adapter.events_by_link(
        [
            {
                "source_link": "finger_tip",
                "contact_point_l": [0.0, 0.0, 0.0],
                "contact_normal_l": [0.0, 0.0, 1.0],
                "contact_force_w": [0.0, 0.0, 2.5],
            }
        ],
        link_poses_w={
            "finger_tip": (
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        },
    )

    assert float(events["finger_tip"].normal_forces[0]) == 2.5


def test_urdf_pressure_pad_grid_metadata_loads_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_tip">
            <pressure_pad name="tip_pad" rows="2" cols="3" point_distance="0.002"
                          taxel_count="6"
                          normal_axis="2" normal_offset="0.001"
                          stiffness="123" damping="7" max_force="4" gamma="0.5" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    specs = load_pressure_pad_specs_from_urdf(urdf)
    maps = load_pressure_taxel_maps_from_urdf(urdf)

    assert len(specs) == 1
    assert specs[0].link_name == "finger_tip"
    assert specs[0].name == "tip_pad"
    assert specs[0].taxel_count == 6
    assert specs[0].origin_semantics == "pad_surface"
    assert maps[0].image_shape == (2, 3)
    assert maps[0].points_l.shape == (6, 3)
    assert maps[0].calibration.stiffness == 123.0
    assert maps[0].calibration.damping == 7.0
    assert maps[0].calibration.max_force == 4.0
    assert maps[0].calibration.gamma == 0.5


def test_urdf_pressure_pad_rectangular_pitch_loads_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad" rows="2" cols="3"
                          row_pitch="0.002" col_pitch="0.006" normal_axis="2" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]
    points = np.asarray(taxel_map.points_l).reshape(2, 3, 3)

    assert spec.point_distance is None
    assert spec.row_distance == 0.002
    assert spec.col_distance == 0.006
    np.testing.assert_allclose(points[:, 0, 0], [-0.001, 0.001], atol=1.0e-9)
    np.testing.assert_allclose(points[0, :, 1], [-0.006, 0.0, 0.006], atol=1.0e-9)


def test_urdf_pressure_pad_size_can_derive_pitch(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad" rows="2" cols="3"
                          pad_size="0.004 0.018" pad_size_semantics="cell_extent"
                          normal_axis="2" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.pad_size == (0.004, 0.018)
    assert spec.pad_size_semantics == "cell_extent"
    assert spec.row_distance == pytest.approx(0.002)
    assert spec.col_distance == pytest.approx(0.006)
    assert taxel_map.image_shape == (2, 3)


def test_urdf_pressure_pad_center_span_size_can_derive_pitch(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad" rows="3" cols="4"
                          pad_size="0.004 0.018" pad_size_semantics="center_span" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]

    assert spec.pad_size_semantics == "center_span"
    assert spec.row_distance == pytest.approx(0.002)
    assert spec.col_distance == pytest.approx(0.006)


def test_urdf_pressure_pad_origin_pose_transforms_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad" rows="1" cols="2" point_distance="0.002" normal_axis="2">
              <origin xyz="0.1 0.2 0.3" rpy="0 0 1.5707963267948966" />
            </pressure_pad>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]
    points = np.asarray(taxel_map.points_l).reshape(1, 2, 3)
    normals = np.asarray(taxel_map.normals_l).reshape(1, 2, 3)

    assert spec.origin_xyz == (0.1, 0.2, 0.3)
    np.testing.assert_allclose(spec.origin_rpy, [0.0, 0.0, np.pi / 2.0], atol=1.0e-12)
    np.testing.assert_allclose(points[0, 0], [0.101, 0.2, 0.3], atol=1.0e-6)
    np.testing.assert_allclose(points[0, 1], [0.099, 0.2, 0.3], atol=1.0e-6)
    np.testing.assert_allclose(normals[0], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], atol=1.0e-6)


def test_urdf_pressure_pad_origin_pose_rotates_normals(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="1" cols="1" point_distance="0.002" normal_axis="2">
              <origin rpy="0 1.5707963267948966 0" />
            </pressure_pad>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    np.testing.assert_allclose(np.asarray(taxel_map.normals_l)[0], [1.0, 0.0, 0.0], atol=1.0e-6)


def test_urdf_pressure_pad_normal_sign_flips_normals(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="1" cols="1" point_distance="0.002" normal_axis="2" normal_sign="-1" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.normal_sign == -1.0
    np.testing.assert_allclose(np.asarray(taxel_map.normals_l)[0], [0.0, 0.0, -1.0], atol=1.0e-6)


def test_urdf_pressure_pad_namespace_attrs_load_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot xmlns:bt="https://example.com/brainco_tactile" name="test_hand">
          <link name="finger_tip">
            <bt:pressure_pad bt:name="tip_pad" bt:rows="2" bt:cols="2" bt:point_distance="0.002">
              <bt:calibration bt:stiffness="321" bt:max_force="6" />
            </bt:pressure_pad>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert taxel_map.link_name == "finger_tip"
    assert taxel_map.image_shape == (2, 2)
    assert taxel_map.calibration.stiffness == 321.0
    assert taxel_map.calibration.max_force == 6.0


def test_urdf_pressure_sensor_tag_loads_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad" />
          <sensor name="mcp_pressure" type="pressure" link="finger_pad"
                  rows="2" cols="3" point_distance="0.002"
                  stiffness="222" max_force="5" />
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.name == "mcp_pressure"
    assert spec.link_name == "finger_pad"
    assert taxel_map.image_shape == (2, 3)
    assert taxel_map.calibration.stiffness == 222.0
    assert taxel_map.calibration.max_force == 5.0


def test_urdf_pressure_pad_child_grid_and_calibration_load_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad">
              <grid h="2" w="3" pitch="0.002" />
              <calibration pressure_stiffness="333" pressure_damping="4"
                           pressure_threshold="0.05" taxel_area="0.000001" />
            </pressure_pad>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.num_rows == 2
    assert spec.num_cols == 3
    assert spec.point_distance == 0.002
    assert taxel_map.image_shape == (2, 3)
    assert taxel_map.calibration.stiffness == 333.0
    assert taxel_map.calibration.damping == 4.0
    assert taxel_map.calibration.threshold == 0.05
    assert taxel_map.calibration.area == 0.000001


def test_urdf_pressure_pad_child_taxels_load_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad name="mcp_pad">
              <taxels grid_shape="2,3" count="6" pitch="0.002" />
            </pressure_pad>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.num_rows == 2
    assert spec.num_cols == 3
    assert spec.taxel_count == 6
    assert spec.point_distance == 0.002
    assert taxel_map.image_shape == (2, 3)


def test_urdf_pressure_pad_child_map_loads_npy_taxel_map(tmp_path):
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    points[..., 1] = 2.0
    normals[..., 2] = 1.0
    np.save(maps_dir / "points.npy", points)
    np.save(maps_dir / "normals.npy", normals)
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad" />
          <pressure_pad source_link="finger_pad" name="mcp_pad">
            <map point_map="maps/points.npy" normal_map="maps/normals.npy"
                 taxel_count="4" correction_scale="0.01" />
          </pressure_pad>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.link_name == "finger_pad"
    assert spec.points_npy is not None and spec.points_npy.name == "points.npy"
    assert spec.normals_npy is not None and spec.normals_npy.name == "normals.npy"
    assert spec.taxel_count == 4
    assert taxel_map.image_shape == (2, 2)
    np.testing.assert_allclose(taxel_map.points_l[:, 1], 0.02)
    np.testing.assert_allclose(taxel_map.normals_l[:, 2], 1.0)


def test_urdf_pressure_pad_resolution_alias_loads_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad taxel_resolution="2x3" taxel_count="6" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.num_rows == 2
    assert spec.num_cols == 3
    assert taxel_map.image_shape == (2, 3)


def test_urdf_pressure_pad_grid_and_calibration_aliases_load_taxel_map(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad num_rows="2" num_cols="3" taxel_pitch="0.002"
                          pressure_gain="2" force_max="8" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    spec = load_pressure_pad_specs_from_urdf(urdf)[0]
    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert spec.num_rows == 2
    assert spec.num_cols == 3
    assert spec.point_distance == 0.002
    assert taxel_map.image_shape == (2, 3)
    assert taxel_map.calibration.gain == 2.0
    assert taxel_map.calibration.max_force == 8.0


def test_urdf_pressure_pad_resolution_conflict_is_rejected(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="2" cols="3" resolution="4x3" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rows=2 conflicts"):
        load_pressure_pad_specs_from_urdf(urdf)


def test_select_pressure_pad_spec_requires_link_when_ambiguous():
    specs = [
        UrdfPressurePadSpec(link_name="right_indexmcp_roll_touch_link", num_rows=2, num_cols=3, point_distance=0.002),
        UrdfPressurePadSpec(link_name="right_indexpip_roll_touch_link", num_rows=2, num_cols=3, point_distance=0.002),
    ]

    assert select_pressure_pad_spec(specs[:1]).link_name == "right_indexmcp_roll_touch_link"
    assert (
        select_pressure_pad_spec(specs, link_name="right_indexpip_roll_touch_link").link_name
        == "right_indexpip_roll_touch_link"
    )
    assert (
        select_pressure_pad_spec(
            [UrdfPressurePadSpec(link_name="right_indexpip_roll_rubber_link", num_rows=2, num_cols=3, point_distance=0.002)],
            link_name="right_indexpip_roll_touch_link",
        ).link_name
        == "right_indexpip_roll_rubber_link"
    )
    assert (
        select_pressure_pad_spec(
            [UrdfPressurePadSpec(link_name="right_ringpip_roll_tubber_link", num_rows=2, num_cols=3, point_distance=0.002)],
            link_name="right_ringpip_roll_touch_link",
        ).link_name
        == "right_ringpip_roll_tubber_link"
    )
    with pytest.raises(ValueError, match="Expected exactly one"):
        select_pressure_pad_spec(specs)


def test_urdf_touch_link_helpers_classify_pressure_candidates(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexpip_roll_touch_link" />
          <link name="right_indexmcp_roll_touch_link" />
          <link name="right_index_tip_touch_link" />
          <link name="right_index_fingertip_touch_link" />
          <link name="right_visual_link" />
        </robot>
        """,
        encoding="utf-8",
    )

    assert load_touch_links_from_urdf(urdf) == [
        "right_indexpip_roll_touch_link",
        "right_indexmcp_roll_touch_link",
        "right_index_tip_touch_link",
        "right_index_fingertip_touch_link",
    ]
    assert load_pressure_touch_links_from_urdf(urdf) == [
        "right_indexpip_roll_touch_link",
        "right_indexmcp_roll_touch_link",
    ]


def test_inspect_pressure_urdf_script_reports_resolution_and_candidates(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" taxel_resolution="2x3" taxel_count="6"
                          point_distance="0.002" origin_semantics="pad_surface"
                          stiffness="123" damping="7" max_force="4" gamma="0.5" />
          </link>
          <link name="right_index_tip_touch_link" />
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run([sys.executable, str(script), str(urdf)], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)

    assert payload["pressure_touch_links"] == ["right_indexmcp_roll_touch_link"]
    assert payload["pressure_pad_count"] == 1
    assert payload["pressure_pads"][0]["name"] == "mcp_pad"
    assert payload["pressure_pads"][0]["link_name"] == "right_indexmcp_roll_touch_link"
    assert payload["pressure_pads"][0]["rows"] == 2
    assert payload["pressure_pads"][0]["cols"] == 3
    assert payload["pressure_pads"][0]["taxel_count"] == 6
    assert payload["pressure_pads"][0]["origin_semantics"] == "pad_surface"
    assert payload["pressure_pads"][0]["point_distance"] == 0.002
    assert payload["pressure_pads"][0]["row_distance"] == 0.002
    assert payload["pressure_pads"][0]["col_distance"] == 0.002
    assert payload["pressure_pads"][0]["pad_size"] is None
    assert payload["pressure_pads"][0]["pad_size_semantics"] == "cell_extent"
    assert payload["pressure_pads"][0]["origin_xyz"] == [0.0, 0.0, 0.0]
    assert payload["pressure_pads"][0]["origin_rpy"] == [0.0, 0.0, 0.0]
    assert payload["pressure_pads"][0]["normal_sign"] == 1.0
    assert payload["pressure_pads"][0]["points_npy"] is None
    assert payload["pressure_pads"][0]["normals_npy"] is None
    assert payload["pressure_pads"][0]["calibration"]["stiffness"] == 123.0
    assert payload["pressure_pads"][0]["calibration"]["damping"] == 7.0
    assert payload["pressure_pads"][0]["calibration"]["max_force"] == 4.0
    assert payload["pressure_pads"][0]["calibration"]["gamma"] == 0.5
    assert payload["pressure_pads_non_surface_origin"] == []
    assert payload["passed"] is True


def test_inspect_pressure_urdf_script_reports_explicit_pad_outside_touch_naming(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_palm_pressure_pad">
            <pressure_pad rows="2" cols="3" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["pressure_touch_links"] == []
    assert payload["pressure_pad_links"] == ["right_palm_pressure_pad"]
    assert payload["pressure_pad_links_not_in_pressure_touch_candidates"] == ["right_palm_pressure_pad"]
    assert payload["passed"] is True


def test_inspect_pressure_urdf_script_accepts_rectangular_pitch_layout(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" rows="2" cols="3" row_pitch="0.002" col_pitch="0.006" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["passed"] is True
    assert payload["pressure_pads"][0]["point_distance"] is None
    assert payload["pressure_pads"][0]["row_distance"] == 0.002
    assert payload["pressure_pads"][0]["col_distance"] == 0.006


def test_inspect_pressure_urdf_script_can_fail_on_missing_layouts(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" taxel_count="6" />
          </link>
          <link name="right_indexpip_roll_touch_link" />
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    assert payload["missing_pressure_pad_links"] == ["right_indexpip_roll_touch_link"]
    assert payload["pressure_pads_missing_layout"] == ["right_indexmcp_roll_touch_link"]


def test_inspect_pressure_urdf_script_flags_duplicate_and_fingertip_pressure_pads(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_a" rows="2" cols="3" point_distance="0.002" />
            <pressure_pad name="mcp_b" rows="2" cols="3" point_distance="0.002" />
          </link>
          <link name="right_index_tip_touch_link">
            <pressure_pad name="tip_pad" rows="2" cols="3" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    assert payload["duplicate_pressure_pad_links"] == ["right_indexmcp_roll_touch_link"]
    assert payload["fingertip_pressure_pad_links"] == ["right_index_tip_touch_link"]


def test_inspect_pressure_urdf_script_flags_taxel_count_mismatch(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" rows="2" cols="3" taxel_count="7" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    assert payload["pressure_pads_taxel_count_mismatch"] == [
        {"link_name": "right_indexmcp_roll_touch_link", "layout_count": 6, "taxel_count": 7}
    ]


def test_inspect_pressure_urdf_script_flags_invalid_grid_layout(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" rows="0" cols="3" point_distance="0.0" normal_axis="9" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    invalid_layout = payload["pressure_pads_invalid_grid_layout"][0]
    assert invalid_layout["link_name"] == "right_indexmcp_roll_touch_link"
    assert invalid_layout["rows"] == 0
    assert invalid_layout["cols"] == 3
    assert invalid_layout["point_distance"] == 0.0
    assert invalid_layout["normal_axis"] == 9
    assert "normal_axis" in invalid_layout["error"]


def test_inspect_pressure_urdf_script_flags_non_surface_origin(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_pad name="mcp_pad" rows="2" cols="3" point_distance="0.002" origin_type="internal" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    assert payload["pressure_pads_non_surface_origin"] == [
        {"link_name": "right_indexmcp_roll_touch_link", "origin_semantics": "pad_internal"}
    ]


def test_inspect_pressure_urdf_script_can_check_npy_layout_files(tmp_path):
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 3, 3), dtype=np.float32)
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "normals.npy", normals)
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts", "--check-layout-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    invalid_layout = payload["pressure_pads_invalid_file_layout"][0]
    assert invalid_layout["link_name"] == "right_indexmcp_roll_touch_link"
    assert invalid_layout["points_npy"].endswith("points.npy")
    assert invalid_layout["normals_npy"].endswith("normals.npy")
    assert "matching" in invalid_layout["error"]


def test_inspect_pressure_urdf_script_base_dir_resolves_npy_layout_files(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    np.save(assets / "points.npy", points)
    np.save(assets / "normals.npy", normals)
    urdf = tmp_path / "urdf" / "hand.urdf"
    urdf.parent.mkdir()
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy" taxel_count="4" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(urdf),
            "--base-dir",
            str(assets),
            "--require-pressure-layouts",
            "--check-layout-files",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["passed"] is True
    assert payload["pressure_pads_invalid_file_layout"] == []
    assert payload["pressure_pads"][0]["points_npy"].endswith("assets/points.npy")


def test_inspect_pressure_urdf_script_flags_npy_taxel_count_mismatch(tmp_path):
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "normals.npy", normals)
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy" taxel_count="5" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [sys.executable, str(script), str(urdf), "--require-pressure-layouts", "--check-layout-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["passed"] is False
    invalid_layout = payload["pressure_pads_invalid_file_layout"][0]
    assert invalid_layout["link_name"] == "right_indexmcp_roll_touch_link"
    assert "taxel_count=5" in invalid_layout["error"]


def test_add_default_pressure_pads_script_writes_pressure_layouts(tmp_path):
    urdf = tmp_path / "hand.urdf"
    out = tmp_path / "hand_pressure.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link" />
          <link name="right_indexmcp_roll_rubber_link" />
          <link name="right_indexpip_roll_touch_link">
            <pressure_pad rows="2" cols="2" point_distance="0.001" />
          </link>
          <link name="right_indexpip_roll_rubber_link" />
          <link name="right_index_tip_touch_link" />
        </robot>
        """,
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "add_default_pressure_pads_to_urdf.py"
    inspect_script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "inspect_pressure_urdf.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(urdf),
            "--out",
            str(out),
            "--rows",
            "3",
            "--cols",
            "4",
            "--point-distance",
            "0.002",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    inspect_result = subprocess.run(
        [sys.executable, str(inspect_script), str(out), "--require-pressure-layouts"],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_payload = json.loads(inspect_result.stdout)
    specs = load_pressure_pad_specs_from_urdf(out)
    root = ET.parse(out).getroot()
    added_material = root.find("./link[@name='right_indexmcp_roll_touch_link']/visual/material")
    added_surface_material = root.find("./link[@name='right_indexmcp_roll_rubber_link']/visual/material")

    assert payload["added_pressure_pad_links"] == ["right_indexmcp_roll_touch_link"]
    assert payload["colored_pressure_pad_links"] == [
        "right_indexmcp_roll_touch_link",
        "right_indexmcp_roll_rubber_link",
        "right_indexpip_roll_touch_link",
        "right_indexpip_roll_rubber_link",
    ]
    assert inspect_payload["passed"] is True
    assert inspect_payload["fingertip_pressure_pad_links"] == []
    assert sorted(spec.link_name for spec in specs) == [
        "right_indexmcp_roll_touch_link",
        "right_indexpip_roll_touch_link",
    ]
    added = next(spec for spec in specs if spec.link_name == "right_indexmcp_roll_touch_link")
    assert added.num_rows == 3
    assert added.num_cols == 4
    assert added.taxel_count == 12
    assert added.point_distance == 0.002
    assert added.origin_semantics == "pad_surface"
    assert added_material is not None
    assert added_material.attrib["name"] == "pressure_pad_cyan"
    assert added_material.find("color").attrib["rgba"] == "0 0.85 1 1"
    assert added_surface_material is not None
    assert added_surface_material.attrib["name"] == "pressure_pad_cyan"
    assert added_surface_material.find("color").attrib["rgba"] == "0 0.85 1 1"


def test_revo21_main_urdf_contains_pressure_layout():
    urdf = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "revo21_right_touch"
        / "urdf"
        / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )

    specs = load_pressure_pad_specs_from_urdf(urdf)
    root = ET.parse(urdf).getroot()
    pressure_links = sorted(spec.link_name for spec in specs)
    visual_links = sorted(
        {
            candidate
            for link_name in pressure_links
            for candidate in (
                link_name,
                link_name.replace("_touch_link", "_rubber_link"),
                link_name.replace("_touch_link", "_tubber_link"),
            )
            if root.find(f"./link[@name='{candidate}']") is not None
        }
    )

    assert len(pressure_links) == 11
    assert len(visual_links) == 21
    assert "right_hand_rubber_link" in pressure_links
    assert "right_midpip_roll_touch_link" in pressure_links
    assert "right_midpip_roll_rubber_link" in visual_links
    assert "right_ringpip_roll_rubber_link" in visual_links
    assert all("tip" not in link_name for link_name in pressure_links)
    assert all(spec.taxel_count == 32 for spec in specs)


def _load_pressure_pad_layout_visualization_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "visualize_pressure_pad_layout.py"
    spec = importlib.util.spec_from_file_location("visualize_pressure_pad_layout", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def test_pressure_pad_layout_visualization_reports_taxel_grid_from_urdf(tmp_path):
    module = _load_pressure_pad_layout_visualization_module()
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="hand">
          <link name="finger_touch_link">
            <pressure_pad rows="2" cols="3" point_distance="0.001" normal_axis="2" origin_semantics="pad_surface" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    report = module.build_layout_report(urdf)
    svg = module.render_layout_svg(report)

    assert report["passed"] is True
    assert report["pad_count"] == 1
    pad = report["pads"][0]
    assert pad["link_name"] == "finger_touch_link"
    assert pad["rows"] == 2
    assert pad["cols"] == 3
    assert pad["taxel_count"] == 6
    assert pad["projection_axes"] == [0, 1]
    assert pad["row_distance_m"] == 0.001
    assert pad["col_distance_m"] == 0.001
    assert pad["origin_xyz_m"] == [0.0, 0.0, 0.0]
    assert pad["origin_rpy_rad"] == [0.0, 0.0, 0.0]
    assert len(pad["taxel_uv_m"]) == 2
    assert len(pad["taxel_uv_m"][0]) == 3
    assert "finger_touch_link" in svg
    assert "pitch=1/1mm" in svg
    assert svg.count("<circle") == 6


def test_pressure_pad_layout_visualization_can_filter_requested_link(tmp_path):
    module = _load_pressure_pad_layout_visualization_module()
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="hand">
          <link name="first_touch_link"><pressure_pad rows="1" cols="1" point_distance="0.001" /></link>
          <link name="second_touch_link"><pressure_pad rows="1" cols="2" point_distance="0.001" /></link>
        </robot>
        """,
        encoding="utf-8",
    )

    report = module.build_layout_report(urdf, links=["second_touch_link"])

    assert report["passed"] is True
    assert [pad["link_name"] for pad in report["pads"]] == ["second_touch_link"]
    assert report["missing_requested_links"] == []


def test_urdf_pressure_pad_origin_semantics_aliases_are_normalized(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="2" cols="2" point_distance="0.002" origin_type="internal" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    specs = load_pressure_pad_specs_from_urdf(urdf)

    assert specs[0].origin_semantics == "pad_internal"


def test_urdf_pressure_pad_rejects_unknown_origin_semantics(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="2" cols="2" point_distance="0.002" origin_semantics="joint_origin" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="origin semantics"):
        load_pressure_pad_specs_from_urdf(urdf)


def test_urdf_pressure_pad_rejects_mismatched_taxel_count(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad rows="2" cols="3" taxel_count="7" point_distance="0.002" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="taxel_count=7"):
        load_pressure_taxel_maps_from_urdf(urdf)


def test_urdf_pressure_pad_taxel_count_alone_is_not_a_layout(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_pad taxel_count="6" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    specs = load_pressure_pad_specs_from_urdf(urdf)

    assert specs[0].taxel_count == 6
    with pytest.raises(ValueError, match="needs either npy files or rows/cols with pitch or pad_size"):
        load_pressure_taxel_maps_from_urdf(urdf)


def test_urdf_pressure_pad_npy_metadata_loads_relative_files(tmp_path):
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    points[..., 0] = 1.0
    normals[..., 2] = 1.0
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "normals.npy", normals)

    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_tip" />
          <gazebo reference="finger_tip">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy"
                                correction_scale="0.01" invert_normals="true">
              <calibration gain="2" bias="0.1" threshold="0.2" />
            </pressure_taxel_map>
          </gazebo>
        </robot>
        """,
        encoding="utf-8",
    )

    taxel_map = load_pressure_taxel_maps_from_urdf(urdf)[0]

    assert taxel_map.link_name == "finger_tip"
    assert taxel_map.image_shape == (2, 2)
    np.testing.assert_allclose(taxel_map.points_l[:, 0], 0.01)
    np.testing.assert_allclose(taxel_map.normals_l[:, 2], -1.0)
    assert taxel_map.calibration.gain == 2.0
    assert taxel_map.calibration.bias == 0.1
    assert taxel_map.calibration.threshold == 0.2


def test_urdf_pressure_pad_npy_rejects_mismatched_taxel_count(tmp_path):
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    np.save(tmp_path / "points.npy", points)
    np.save(tmp_path / "normals.npy", normals)
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy" taxel_count="5" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="taxel_count=5"):
        load_pressure_taxel_maps_from_urdf(urdf)


def test_urdf_pressure_pad_missing_npy_files_respects_require_files(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_pad">
            <pressure_taxel_map points_npy="missing_points.npy" normals_npy="missing_normals.npy" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    specs = load_pressure_pad_specs_from_urdf(urdf, require_files=False)

    assert specs[0].points_npy.name == "missing_points.npy"
    with pytest.raises(FileNotFoundError, match="missing_points.npy"):
        load_pressure_pad_specs_from_urdf(urdf)


def test_pressure_calibration_file_supports_per_taxel_arrays(tmp_path):
    max_force = np.full((2, 3), 100.0, dtype=np.float32)
    np.save(tmp_path / "max_force.npy", max_force)
    calib_path = tmp_path / "pressure_calib.json"
    calib_path.write_text(
        json.dumps(
            {
                "calibration": {
                    "gain": [[1, 2, 3], [4, 5, 6]],
                    "stiffness": 2.0,
                    "max_force_npy": "max_force.npy",
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = load_pressure_calibration_overrides(calib_path, image_shape=(2, 3))

    assert overrides["stiffness"] == 2.0
    np.testing.assert_allclose(overrides["gain"], np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    np.testing.assert_allclose(overrides["max_force"], np.full((6,), 100.0, dtype=np.float32))

    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=3,
        point_distance=1.0,
        normal_axis=2,
        calibration=PressureCalibration(
            gain=overrides["gain"],
            stiffness=overrides["stiffness"],
            max_force=overrides["max_force"],
        ),
    )
    output = taxel_map.force_from_penetration(np.ones((2, 3), dtype=np.float32))

    np.testing.assert_allclose(
        output.raw_force_map,
        np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32),
    )


def test_pressure_trace_report_detects_precontact_leakage():
    pressure = np.zeros((3, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[0, 0, 0, 0] = 1.0
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
    }

    report = pressure_trace_report(trace, active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report, precontact_leakage_threshold=0.0)

    assert "reference" not in report
    assert "reference_layer" not in report
    assert report["precontact_leakage_fraction"] > 0.0
    assert report["precontact_active_steps"] == 1
    assert evaluation["passed"] is False


def test_validate_pressure_trace_v1_accepts_l0_trace():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=3,
        num_cols=4,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=3, indentation_start_m=0.0, indentation_end_m=0.001),
    )

    result = validate_pressure_trace_v1(trace.arrays | {"metadata_json": np.asarray(json.dumps(trace.metadata))})

    assert result["passed"] is True
    assert result["warnings"] == []
    assert result["schema"] == "pressure_trace_v1"
    assert result["sensor_count"] == 1
    assert result["image_shape"] == [3, 4]
    assert result["metadata"]["pressure_backend_id"] == "l0_analytic"


def test_validate_pressure_trace_v1_reads_metadata_sidecar(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=2,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=2, indentation_start_m=0.0, indentation_end_m=0.001),
    )
    trace_path = tmp_path / "trace.npz"
    metadata_path = tmp_path / "trace.metadata.json"
    np.savez_compressed(trace_path, **trace.arrays)
    metadata_path.write_text(json.dumps(trace.metadata), encoding="utf-8")

    result = validate_pressure_trace_v1(trace_path)

    assert result["passed"] is True
    assert result["warnings"] == []
    assert result["metadata"]["pressure_trace_schema_version"] == "pressure_trace_v1"
    assert result["metadata"]["pressure_backend_id"] == "l0_analytic"


def test_verify_pressure_trace_cli_fails_unsafe_reference_layer_override(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=2,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=3, indentation_start_m=0.0, indentation_end_m=0.001),
    )
    trace_path, _metadata_path = trace.save(tmp_path, run_id="cli_gate")
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "verify_pressure_trace.py"
    base_cmd = [
        sys.executable,
        str(script),
        str(trace_path),
        "--reference-key",
        "analytic_depth_m",
        "--reference-layer",
        "L1_model_reference",
        "--fail-on-threshold",
    ]

    blocked = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
    allowed = subprocess.run(
        base_cmd + ["--allow-reference-layer-override"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode == 2
    blocked_checks = {check["name"]: check for check in json.loads(blocked.stdout)["evaluation"]["checks"]}
    assert blocked_checks["reference.layer_override_is_safe"]["passed"] is False
    assert allowed.returncode == 0


def test_verify_pressure_trace_cli_requires_explicit_l3_dense_diagnostic_layer(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=2,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=3, indentation_start_m=0.0, indentation_end_m=0.001),
    )
    trace_path = tmp_path / "l3_cli_gate.npz"
    metadata_path = tmp_path / "l3_cli_gate.metadata.json"
    np.savez_compressed(trace_path, **(trace.arrays | {"real_pad_pressure_n": trace.arrays["penetration_m"]}))
    metadata_path.write_text(json.dumps(trace.metadata), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "verify_pressure_trace.py"
    base_cmd = [
        sys.executable,
        str(script),
        str(trace_path),
        "--reference-key",
        "real_pad_pressure_n",
        "--fail-on-threshold",
    ]

    blocked = subprocess.run(base_cmd, check=False, capture_output=True, text=True)
    allowed = subprocess.run(
        base_cmd + ["--dense-reference-layer", "L3_real_calibration"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode == 2
    blocked_checks = {check["name"]: check for check in json.loads(blocked.stdout)["evaluation"]["checks"]}
    assert blocked_checks["reference.layer_allows_dense_acceptance"]["passed"] is False
    assert allowed.returncode == 0


def test_verify_pressure_trace_cli_reports_contract_error_for_malformed_trace(tmp_path):
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace_path = tmp_path / "bad_trace.npz"
    np.savez_compressed(
        trace_path,
        penetration_m=pressure,
        signed_distance_m=pressure,
        penetration_velocity_mps=pressure,
        pressure_raw_n=pressure,
        pressure_norm=np.zeros((2,), dtype=np.float32),
        total_force_n=np.zeros((2, 1), dtype=np.float32),
        center_of_pressure_px=np.zeros((2, 1, 2), dtype=np.float32),
        pressure_taxel_points_l_m=np.zeros((1, 2, 2, 3), dtype=np.float32),
        pressure_taxel_normals_l=np.zeros((1, 2, 2, 3), dtype=np.float32),
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "verify_pressure_trace.py"
    out_json = tmp_path / "reports" / "bad_trace.json"

    result = subprocess.run(
        [sys.executable, str(script), str(trace_path), "--out-json", str(out_json), "--fail-on-threshold"],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    written_payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert result.returncode == 2
    assert written_payload == payload
    assert payload["contract"]["passed"] is False
    assert any("pressure_norm must have shape" in error for error in payload["contract"]["errors"])
    assert payload["report"]["available"] is False


def _load_watertight_mesh_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "generate_pressure_watertight_mesh.py"
    spec = importlib.util.spec_from_file_location("generate_pressure_watertight_mesh", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def test_validate_pressure_trace_v1_rejects_bad_layout_shape():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 3, 3, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "metadata_json": np.asarray(json.dumps({"pressure_trace_schema_version": "pressure_trace_v1"})),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is False
    assert any("pressure_taxel_points_l_m" in error for error in result["errors"])
    assert any("pressure_backend_id" in warning for warning in result["warnings"])


def test_validate_pressure_trace_v1_reports_bad_core_array_rank():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": np.zeros((2,), dtype=np.float32),
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is False
    assert any("pressure_norm must have shape" in error for error in result["errors"])


def test_validate_pressure_trace_v1_reports_bad_required_tshw_rank():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": np.zeros((2,), dtype=np.float32),
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is False
    assert any("penetration_m must have shape" in error for error in result["errors"])


def test_validate_pressure_trace_v1_reports_bad_optional_tshw_rank():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "normal_ray_penetration_m": np.zeros((2,), dtype=np.float32),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is False
    assert any("normal_ray_penetration_m must have shape" in error for error in result["errors"])


def test_validate_pressure_trace_v1_rejects_unsupported_schema_version():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "metadata_json": np.asarray(json.dumps({"pressure_trace_schema_version": "pressure_trace_v2"})),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is False
    assert "unsupported pressure_trace_schema_version 'pressure_trace_v2'" in result["errors"]


def test_validate_pressure_trace_v1_reports_bad_metadata_json():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "metadata_json": np.asarray("{not-json"),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is True
    assert "metadata_json parse failed" in result["warnings"]
    assert result["metadata"]["metadata_json_parse_error"] is True


def test_validate_pressure_trace_v1_reports_non_object_metadata_json():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "metadata_json": np.asarray("[1, 2, 3]"),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is True
    assert "metadata_json is not an object" in result["warnings"]
    assert result["metadata"]["metadata_json_not_object"] is True


def test_validate_pressure_trace_v1_reports_non_scalar_metadata_json_array():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": pressure,
        "signed_distance_m": pressure,
        "penetration_velocity_mps": pressure,
        "pressure_raw_n": pressure,
        "pressure_norm": pressure,
        "total_force_n": np.zeros((2, 1), dtype=np.float32),
        "center_of_pressure_px": np.zeros((2, 1, 2), dtype=np.float32),
        "pressure_taxel_points_l_m": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "pressure_taxel_normals_l": np.zeros((1, 2, 2, 3), dtype=np.float32),
        "metadata_json": np.asarray(["{}"]),
    }

    result = validate_pressure_trace_v1(trace)

    assert result["passed"] is True
    assert "metadata_json is not an object" in result["warnings"]
    assert result["metadata"]["metadata_json_not_object"] is True


def test_validate_pressure_trace_v1_reports_bad_metadata_sidecar(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=2,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=2, indentation_start_m=0.0, indentation_end_m=0.001),
    )
    trace_path = tmp_path / "trace.npz"
    metadata_path = tmp_path / "trace.metadata.json"
    np.savez_compressed(trace_path, **(trace.arrays | {"metadata_json": np.asarray(json.dumps(trace.metadata))}))
    metadata_path.write_text("{not-json", encoding="utf-8")

    result = validate_pressure_trace_v1(trace_path)

    assert result["passed"] is True
    assert "metadata sidecar parse failed" in result["warnings"]
    assert result["metadata"]["metadata_file_parse_error"] is True


def test_validate_pressure_trace_v1_reports_non_object_metadata_sidecar(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=2,
        num_cols=2,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=10.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.002),
        AnalyticPressTrajectory(steps=2, indentation_start_m=0.0, indentation_end_m=0.001),
    )
    trace_path = tmp_path / "trace.npz"
    metadata_path = tmp_path / "trace.metadata.json"
    np.savez_compressed(trace_path, **(trace.arrays | {"metadata_json": np.asarray(json.dumps(trace.metadata))}))
    metadata_path.write_text("[1, 2, 3]", encoding="utf-8")

    result = validate_pressure_trace_v1(trace_path)

    assert result["passed"] is True
    assert "metadata sidecar is not an object" in result["warnings"]
    assert result["metadata"]["metadata_file_not_object"] is True


def test_pressure_trace_report_compares_reference_masks():
    pressure = np.zeros((3, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    reference = np.zeros((3, 1, 4, 4), dtype=np.float32)
    pressure[1, 0, 0, 0] = 1.0
    penetration[1, 0, 0, 0] = 0.001
    reference[1, 0, 0:2, 0:2] = 0.001
    pressure[2, 0, 0, 1] = 1.0
    penetration[2, 0, 0, 1] = 0.001
    reference[2, 0, 0:2, 2:4] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "tacmap_raw_m": reference,
    }

    report = pressure_trace_report(trace, reference_key="tacmap_raw_m", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["precontact_leakage_fraction"] == 0.0
    assert report["reference_key"] == "tacmap_raw_m"
    assert report["reference_layer"] == "L1_model_reference"
    assert report["reference"]["reference_layer"] == "L1_model_reference"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    assert report["reference"]["centroid_error_px_max"] == 0.0
    assert report["reference"]["bbox_error_px_max"] == 0.0
    assert report["reference"]["depth_rmse_m_max"] == 0.0
    assert report["reference"]["offset_error_frames_by_sensor"] == [0]
    assert evaluation["passed"] is True


def test_reference_tensor_shape_failure_is_reported_not_raised():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    pressure[1, 0, 0, 0] = 1.0
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": pressure * 0.001,
        "geometry_normal_ray_sample_penetrations_m": np.zeros((2, 1, 2, 2, 3), dtype=np.float32),
    }

    report = pressure_trace_report(
        trace,
        reference_key="geometry_normal_ray_sample_penetrations_m",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference"]["available"] is False
    assert report["reference"]["reference_shape"] == [2, 1, 2, 2, 3]
    available_check = next(check for check in evaluation["checks"] if check["name"] == "reference.available")
    assert available_check["passed"] is False
    assert evaluation["passed"] is False


def test_reference_alignment_coverage_flags_invalid_contact_region():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    reference = np.zeros_like(pressure)
    valid_mask = np.array([[[1, 1], [1, 0]]], dtype=np.uint8)
    pressure[1, 0, 0, 0] = 1.0
    pressure[1, 0, 1, 1] = 1.0
    penetration[1, 0, 0, 0] = 0.001
    penetration[1, 0, 1, 1] = 0.001
    reference[1, 0, 0, 0] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "tacmap_raw_aligned_m": reference,
        "tacmap_raw_aligned_m_valid_mask": valid_mask,
    }

    report = pressure_trace_report(
        trace,
        reference_key="tacmap_raw_aligned_m",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(report, reference_iou_threshold=0.0)

    assert report["reference"]["reference_valid_mask_key"] == "tacmap_raw_aligned_m_valid_mask"
    assert report["reference"]["alignment_valid_taxel_fraction_min"] == 0.75
    assert report["reference"]["alignment_contact_region_valid_fraction_min"] == 0.5
    assert report["reference"]["alignment_invalid_contact_fraction_max"] == 0.5
    assert report["reference"]["alignment_invalid_zero_fill_fraction_max"] == 0.25
    contact_check = next(
        check
        for check in evaluation["checks"]
        if check["name"] == "reference.alignment_contact_region_valid_fraction_min"
    )
    invalid_check = next(
        check
        for check in evaluation["checks"]
        if check["name"] == "reference.alignment_invalid_contact_fraction_max"
    )
    assert contact_check["passed"] is False
    assert invalid_check["passed"] is False
    assert evaluation["passed"] is False


def test_normal_ray_reference_uses_alignment_valid_mask_fallback():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    reference = np.zeros_like(pressure)
    valid_mask = np.array([[[1, 0], [1, 1]]], dtype=np.uint8)
    pressure[1, 0, 0, 1] = 1.0
    reference[1, 0, 0, 1] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": reference,
        "normal_ray_penetration_m": reference,
        "normal_ray_alignment_valid_mask": valid_mask,
    }

    report = pressure_trace_report(trace, reference_key="normal_ray_penetration_m")

    assert report["reference"]["reference_valid_mask_key"] == "normal_ray_alignment_valid_mask"
    assert report["reference"]["alignment_contact_region_valid_fraction_min"] == 0.0


def test_pressure_trace_report_adds_reference_origin_alignment_diagnostics():
    pressure = np.zeros((1, 1, 2, 2), dtype=np.float32)
    reference = np.zeros_like(pressure)
    pressure[0, 0, 0, 0] = 1.0
    reference[0, 0, 0, 0] = 0.001
    points = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.001, 0.0]],
                [[0.0, 0.0, 0.001], [0.0, 0.001, 0.001]],
            ]
        ],
        dtype=np.float32,
    )
    normals = np.zeros_like(points)
    normals[..., 0] = 1.0
    grid_points = points.copy()
    grid_points[..., 0] -= 0.01
    grid_points[..., 1] += 0.0001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": reference,
        "normal_ray_penetration_m": reference,
        "pressure_taxel_points_l_m": points,
        "pressure_taxel_normals_l": normals,
        "tacmap_grid_points_l_m": grid_points,
        "normal_ray_alignment_source_index": np.array([[[0, 1], [2, 3]]], dtype=np.int64),
        "normal_ray_alignment_valid_mask": np.ones((1, 2, 2), dtype=np.uint8),
        "normal_ray_alignment_nn_distance_m": np.full((1, 2, 2), 0.0001, dtype=np.float32),
    }

    report = pressure_trace_report(trace, reference_key="normal_ray_penetration_m")
    direct = pressure_reference_origin_alignment_diagnostics(trace)
    frame_diagnostics = pressure_reference_frame_diagnostics(trace, reference_key="normal_ray_penetration_m")
    alignment = report["reference"]["origin_alignment"]

    assert alignment["available"] is True
    assert report["reference_origin_alignment"]["available"] is True
    assert frame_diagnostics["summary"]["origin_alignment"]["available"] is True
    assert alignment["valid_taxels"] == 4
    assert alignment["valid_fraction"] == 1.0
    assert alignment["normal_delta_m_mean"] == pytest.approx(-0.01)
    assert alignment["abs_normal_delta_m_mean"] == pytest.approx(0.01)
    assert alignment["tangent_delta_m_mean"] == pytest.approx(0.0001)
    assert alignment["nn_distance_m_max"] == pytest.approx(0.0001)
    assert direct["normal_delta_m_mean"] == alignment["normal_delta_m_mean"]
    assert frame_diagnostics["summary"]["origin_alignment"]["normal_delta_m_mean"] == alignment["normal_delta_m_mean"]


def test_pressure_reference_frame_diagnostics_separates_boundary_error_types():
    pressure = np.zeros((3, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    reference = np.zeros_like(pressure)
    pressure[0, 0, 0, 0] = 1.0
    penetration[0, 0, 0, 0] = 0.001
    reference[1, 0, 0, 1] = 0.002
    pressure[2, 0, 1, 1] = 1.0
    penetration[2, 0, 1, 1] = 0.003
    reference[2, 0, 1, 1] = 0.003
    trace = {
        "pressure_norm": pressure,
        "penetration_m": penetration,
        "normal_ray_penetration_m": reference,
    }

    diagnostics = pressure_reference_frame_diagnostics(
        trace,
        reference_key="normal_ray_penetration_m",
        top_k=3,
    )
    frames = diagnostics["frames"]
    summary = diagnostics["summary"]

    assert summary["reference_layer"] == "L1_model_reference"
    assert summary["false_positive_taxels_total"] == 1
    assert summary["false_negative_taxels_total"] == 1
    assert summary["frames_with_false_positive"] == 1
    assert summary["frames_with_false_negative"] == 1
    assert summary["pressure_onset_step_by_sensor"] == [0]
    assert summary["reference_onset_step_by_sensor"] == [1]
    assert frames[0]["classification"] == "false_positive_only"
    assert frames[1]["classification"] == "false_negative_only"
    assert frames[2]["classification"] == "perfect_overlap"
    assert summary["worst_iou_frames"][0]["iou"] == 0.0


def test_pressure_reference_frame_diagnostics_reports_non_dense_reference():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    samples = np.zeros((2, 1, 2, 2, 3), dtype=np.float32)
    trace = {
        "pressure_norm": pressure,
        "penetration_m": pressure,
        "geometry_normal_ray_sample_penetrations_m": samples,
    }

    diagnostics = pressure_reference_frame_diagnostics(
        trace,
        reference_key="geometry_normal_ray_sample_penetrations_m",
    )

    assert diagnostics["summary"]["available"] is False
    assert diagnostics["summary"]["reference_shape"] == [2, 1, 2, 2, 3]
    assert diagnostics["summary"]["reference_layer"] == "benchmark_reference"
    assert diagnostics["frames"] == []


def test_pressure_reference_frame_diagnostics_no_overlap_uses_summary_shape():
    pressure = np.zeros((1, 0, 2, 2), dtype=np.float32)
    trace = {
        "pressure_norm": pressure,
        "penetration_m": pressure,
        "normal_ray_penetration_m": pressure,
    }

    diagnostics = pressure_reference_frame_diagnostics(
        trace,
        reference_key="normal_ray_penetration_m",
    )

    assert diagnostics["summary"]["available"] is False
    assert diagnostics["summary"]["reason"] == "no overlapping steps or sensors"
    assert diagnostics["summary"]["reference_layer"] == "L1_model_reference"
    assert diagnostics["frames"] == []


def test_pressure_sample_boundary_diagnostics_groups_raw_samples():
    pressure = np.zeros((1, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    reference = np.zeros_like(pressure)
    samples = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)

    pressure[0, 0, 0, 0] = 1.0
    penetration[0, 0, 0, 0] = 0.001
    reference[0, 0, 0, 0] = 0.002
    samples[0, 0, 0, 0] = np.array([0.001, 0.001, 0.0], dtype=np.float32)

    pressure[0, 0, 0, 1] = 1.0
    penetration[0, 0, 0, 1] = 0.0005
    samples[0, 0, 0, 1] = np.array([0.0005, 0.0, 0.0], dtype=np.float32)

    reference[0, 0, 1, 0] = 0.003
    trace = {
        "geometry_normal_ray_pressure_norm": pressure,
        "geometry_normal_ray_penetration_m": penetration,
        "normal_ray_penetration_m": reference,
        "geometry_normal_ray_sample_penetrations_m": samples,
    }

    diagnostics = pressure_sample_boundary_diagnostics(trace, top_k=2)
    summary = diagnostics["summary"]
    groups = summary["groups"]

    assert summary["sample_count"] == 3
    assert summary["reference_layer"] == "L1_model_reference"
    assert groups["true_positive"]["taxels"] == 1
    assert groups["false_positive"]["taxels"] == 1
    assert groups["false_negative"]["taxels"] == 1
    np.testing.assert_allclose(groups["true_positive"]["sample_support_fraction_mean"], 2.0 / 3.0)
    np.testing.assert_allclose(groups["false_positive"]["sample_support_fraction_mean"], 1.0 / 3.0)
    np.testing.assert_allclose(groups["false_negative"]["sample_support_fraction_mean"], 0.0)
    np.testing.assert_allclose(groups["false_positive"]["sample_max_penetration_m_mean"], 0.0005)
    np.testing.assert_allclose(groups["false_negative"]["reference_depth_m_mean"], 0.003)
    assert diagnostics["frames"][0]["classification"] == "mixed_boundary_error"
    assert summary["largest_false_positive_frames"][0]["false_positive_taxels"] == 1
    assert summary["largest_false_negative_frames"][0]["false_negative_taxels"] == 1


def test_pressure_sample_boundary_diagnostics_reports_non_dense_reference():
    pressure = np.zeros((1, 1, 2, 2), dtype=np.float32)
    samples = np.zeros((1, 1, 2, 2, 3), dtype=np.float32)
    trace = {
        "geometry_normal_ray_pressure_norm": pressure,
        "geometry_normal_ray_penetration_m": pressure,
        "geometry_normal_ray_sample_penetrations_m": samples,
    }

    diagnostics = pressure_sample_boundary_diagnostics(
        trace,
        reference_key="geometry_normal_ray_sample_penetrations_m",
    )

    assert diagnostics["summary"]["available"] is False
    assert diagnostics["summary"]["reference_shape"] == [1, 1, 2, 2, 3]
    assert diagnostics["summary"]["reference_layer"] == "benchmark_reference"
    assert diagnostics["frames"] == []


def test_pressure_reference_layer_inference_keeps_model_reference_distinct_from_gt():
    assert infer_pressure_reference_layer(None) is None
    assert infer_pressure_reference_layer("analytic_depth_m") == "L0_analytic"
    assert infer_pressure_reference_layer("analytic_penetration_m") == "L0_analytic"
    assert infer_pressure_reference_layer("analytic_contact_mask") == "benchmark_reference"
    assert infer_pressure_reference_layer("analytic_signed_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("tacmap_raw_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("tacmap_contact_depth_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("normal_ray_penetration_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("normal_ray_contact_penetration_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("normal_ray_penetration_velocity_mps") == "benchmark_reference"
    assert infer_pressure_reference_layer("normal_ray_pressure_norm") == "benchmark_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_pressure_norm") == "benchmark_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_sample_penetrations_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_sample_mean_penetration_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_sample_positive_mean_penetration_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_sample_max_penetration_m") == "L1_model_reference"
    assert infer_pressure_reference_layer("normal_ray_signed_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("normal_ray_alignment_nn_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("geometry_normal_ray_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("uipc_signed_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("physx_contact_count") == "L1_sparse_sanity"
    assert infer_pressure_reference_layer("contact_force_map") == "L1_sparse_sanity"
    assert infer_pressure_reference_layer("contact_mask") == "benchmark_reference"
    assert infer_pressure_reference_layer("uipc_penetration_m") == "L2_offline_oracle"
    assert infer_pressure_reference_layer("uipc_penetration_velocity_mps") == "benchmark_reference"
    assert infer_pressure_reference_layer("uipc_contact_penetration_m") == "L2_offline_oracle"
    assert infer_pressure_reference_layer("hydroelastic_depth_m") == "L2_offline_oracle"
    assert infer_pressure_reference_layer("uipc_distance_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("uipc_pressure_mpa") == "benchmark_reference"
    assert infer_pressure_reference_layer("real_pad_pressure_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("real_pad_contact_pressure_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("tekscan_pressure_map_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("pps_pressure_array_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("pressure_film_pressure_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("load_cell_force_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("gelsight_force_map_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("digit_pressure_map_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("sparsh_tacbench_force_label_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("feelanyforce_force_vector_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("vision_tactile_pressure_estimate_n") == "L3_real_calibration"
    assert infer_pressure_reference_layer("visuo_tactile_force_field_n") == "benchmark_reference"
    assert infer_pressure_reference_layer("visuotactile_force_field_n") == "benchmark_reference"
    assert infer_pressure_reference_layer("vbts_deform") == "benchmark_reference"
    assert infer_pressure_reference_layer("gelsight_fem_depth_m") == "L3_real_calibration"
    assert infer_pressure_reference_layer("sparsh_hydroelastic_depth_m") == "L3_real_calibration"
    assert infer_pressure_reference_layer("feelanyforce_uipc_penetration_m") == "L3_real_calibration"
    assert infer_pressure_reference_layer("tacsl_sdf_penetration_m") == "benchmark_reference"
    assert infer_pressure_reference_layer("feats_fem_depth_m") == "L3_real_calibration"
    assert infer_pressure_reference_layer("vendor_reference_map") == "benchmark_reference"


def test_sparse_reference_layer_cannot_drive_dense_acceptance_by_default():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 1, 1] = 1.0
    penetration[1, 0, 1, 1] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "physx_contact_force_map": penetration,
    }

    report = pressure_trace_report(trace, reference_key="physx_contact_force_map", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference_layer"] == "L1_sparse_sanity"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is False
    assert evaluation["passed"] is False


def test_reference_layer_can_be_explicitly_allowed_for_dense_diagnostics():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 0, 1] = 1.0
    penetration[1, 0, 0, 1] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "physx_contact_force_map": penetration,
    }

    report = pressure_trace_report(trace, reference_key="physx_contact_force_map", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report, dense_reference_layers=("L1_sparse_sanity",))

    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is True
    assert evaluation["passed"] is True


def test_l2_offline_oracle_can_drive_dense_acceptance_by_default():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 1, 0] = 1.0
    penetration[1, 0, 1, 0] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "uipc_penetration_m": penetration,
    }

    report = pressure_trace_report(trace, reference_key="uipc_penetration_m", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference_layer"] == "L2_offline_oracle"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is True
    assert evaluation["passed"] is True


def test_unsafe_reference_layer_override_is_rejected_by_default():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 1, 0] = 1.0
    penetration[1, 0, 1, 0] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "physx_contact_force_map": penetration,
    }

    report = pressure_trace_report(
        trace,
        reference_key="physx_contact_force_map",
        reference_layer="L1_model_reference",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(report)
    override_allowed = evaluate_pressure_trace_report(report, allow_reference_layer_override=True)

    assert report["inferred_reference_layer"] == "L1_sparse_sanity"
    assert report["reference_layer"] == "L1_model_reference"
    assert report["reference_layer_override_conflict"] is True
    override_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_override_is_safe")
    assert override_check["passed"] is False
    assert evaluation["passed"] is False
    assert override_allowed["passed"] is True


def test_benchmark_reference_override_is_rejected_by_default():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    pressure[1, 0, 0, 1] = 1.0
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": pressure * 0.001,
        "vendor_reference_map": pressure * 0.001,
    }

    report = pressure_trace_report(
        trace,
        reference_key="vendor_reference_map",
        reference_layer="L1_model_reference",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(report)

    assert report["inferred_reference_layer"] == "benchmark_reference"
    assert report["reference_layer_override_conflict"] is True
    override_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_override_is_safe")
    assert override_check["passed"] is False
    assert evaluation["passed"] is False


def test_dense_reference_layer_override_is_rejected_by_default():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    pressure[1, 0, 0, 0] = 1.0
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": pressure * 0.001,
        "analytic_depth_m": pressure * 0.001,
    }

    report = pressure_trace_report(
        trace,
        reference_key="analytic_depth_m",
        reference_layer="L1_model_reference",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(report)
    override_allowed = evaluate_pressure_trace_report(report, allow_reference_layer_override=True)

    assert report["inferred_reference_layer"] == "L0_analytic"
    assert report["reference_layer_override_conflict"] is True
    override_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_override_is_safe")
    assert override_check["passed"] is False
    assert evaluation["passed"] is False
    assert override_allowed["passed"] is True


def test_real_calibration_layer_is_not_default_dense_sim_gt():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 0, 0] = 1.0
    penetration[1, 0, 0, 0] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "real_pad_pressure_n": penetration,
    }

    report = pressure_trace_report(trace, reference_key="real_pad_pressure_n", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference_layer"] == "L3_real_calibration"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is False
    assert evaluation["passed"] is False


def test_vision_tactile_reference_layer_is_not_default_dense_sim_gt():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 0, 1] = 1.0
    penetration[1, 0, 0, 1] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "gelsight_force_map_n": penetration,
    }

    report = pressure_trace_report(trace, reference_key="gelsight_force_map_n", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference_layer"] == "L3_real_calibration"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is False
    assert evaluation["passed"] is False


def test_model_sensor_reference_layer_is_not_default_dense_sim_gt():
    pressure = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration = np.zeros_like(pressure)
    pressure[1, 0, 1, 0] = 1.0
    penetration[1, 0, 1, 0] = 0.001
    trace = {
        "pressure_norm": pressure,
        "pressure_raw_n": pressure,
        "penetration_m": penetration,
        "tacsl_sdf_penetration_m": penetration,
    }

    report = pressure_trace_report(trace, reference_key="tacsl_sdf_penetration_m", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(report)

    assert report["reference_layer"] == "benchmark_reference"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    layer_check = next(check for check in evaluation["checks"] if check["name"] == "reference.layer_allows_dense_acceptance")
    assert layer_check["passed"] is False
    assert evaluation["passed"] is False


def test_l0_square_trace_has_expected_center_and_total_force():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=5,
        num_cols=5,
        point_distance=1.0,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=2.0, max_force=10.0),
    )

    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=1.1),
        AnalyticPressTrajectory(steps=1, indentation_end_m=1.0),
    )

    pressure = trace.arrays["pressure_raw_n"][0, 0]
    assert pressure[2, 2] == 2.0
    assert np.count_nonzero(pressure) == 1
    np.testing.assert_allclose(trace.arrays["total_force_n"], np.array([[2.0]], dtype=np.float32))
    np.testing.assert_allclose(trace.arrays["center_of_pressure_px"], np.array([[[2.0, 2.0]]], dtype=np.float32))


def test_l0_sphere_footprint_expands_monotonically_with_depth():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=9,
        num_cols=9,
        point_distance=0.001,
        normal_axis=2,
    )
    uv = taxel_map.points_l[:, :2]
    presser = AnalyticPresserSpec(kind="sphere", radius_m=0.004)

    shallow = analytic_presser_penetration(uv, presser, indentation_m=0.00025)
    deep = analytic_presser_penetration(uv, presser, indentation_m=0.001)

    assert np.count_nonzero(deep > 0.0) > np.count_nonzero(shallow > 0.0)
    assert float(np.max(deep)) > float(np.max(shallow))


def test_l0_trace_is_compatible_with_pressure_trace_verifier():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=5,
        num_cols=5,
        point_distance=1.0,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=1.0, max_force=10.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="cylinder", size_m=3.1),
        AnalyticPressTrajectory(steps=4, indentation_start_m=0.0, indentation_end_m=1.0),
    )

    report = pressure_trace_report(trace.arrays, reference_key="analytic_depth_m", active_threshold=1.0e-6)
    evaluation = evaluate_pressure_trace_report(
        report,
        reference_iou_threshold=1.0,
        centroid_error_threshold_px=0.0,
        bbox_error_threshold_px=0.0,
        depth_rmse_threshold_m=0.0,
    )

    assert report["precontact_leakage_fraction"] == 0.0
    assert report["reference_layer"] == "L0_analytic"
    assert report["reference"]["active_mask_iou_min"] == 1.0
    assert report["reference"]["bbox_error_px_max"] == 0.0
    assert report["reference"]["depth_rmse_m_max"] == 0.0
    assert evaluation["passed"] is True


def test_warpsdf_penetration_source_converts_signed_distance():
    source = WarpSdfPenetrationSource(contact_deadband_m=0.0001)
    frame = source.frame_from_signed_distance(
        np.array([[[[0.001, -0.00005], [-0.0002, -0.001]]]], dtype=np.float32)
    )

    np.testing.assert_allclose(
        frame.penetration_m,
        np.array([[[[0.0, 0.0], [0.0001, 0.0009]]]], dtype=np.float32),
        atol=1.0e-8,
    )
    assert frame.contact_mask.tolist() == [[[[False, False], [True, True]]]]


def test_normal_ray_penetration_source_converts_hit_distance():
    source = NormalRayPenetrationSource(rest_distance_m=0.01, contact_deadband_m=0.001)
    frame = source.frame_from_ray_distance(
        np.array([[[[0.012, 0.009], [0.006, 0.004]]]], dtype=np.float32),
        valid_mask=np.array([[[[True, True], [False, True]]]], dtype=bool),
    )

    np.testing.assert_allclose(
        frame.penetration_m,
        np.array([[[[0.0, 0.0], [0.0, 0.005]]]], dtype=np.float32),
        atol=1.0e-8,
    )
    assert frame.contact_mask.tolist() == [[[[False, False], [False, True]]]]


def test_normal_ray_penetration_source_treats_inf_hit_as_miss():
    source = NormalRayPenetrationSource(rest_distance_m=0.01)
    frame = source.frame_from_ray_distance(
        np.array([[[[np.inf, 0.006]]]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        frame.penetration_m,
        np.array([[[[0.0, 0.004]]]], dtype=np.float32),
        atol=1.0e-8,
    )
    assert frame.contact_mask.tolist() == [[[[False, True]]]]


def test_surface_gap_proxy_activates_without_signed_penetration():
    signed_source = WarpSdfPenetrationSource()
    gap_source = NormalRayPenetrationSource(rest_distance_m=0.001)

    signed_frame = signed_source.frame_from_signed_distance(np.array([[[[0.0004]]]], dtype=np.float32))
    gap_frame = gap_source.frame_from_ray_distance(np.array([[[[0.0004]]]], dtype=np.float32))

    np.testing.assert_allclose(signed_frame.penetration_m, np.array([[[[0.0]]]], dtype=np.float32))
    np.testing.assert_allclose(gap_frame.penetration_m, np.array([[[[0.0006]]]], dtype=np.float32))
    assert signed_frame.contact_mask.tolist() == [[[[False]]]]
    assert gap_frame.contact_mask.tolist() == [[[[True]]]]


def test_geometry_normal_ray_source_sphere_footprint_has_center_maximum():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=3,
        num_cols=3,
        point_distance=0.002,
        normal_axis=2,
    )
    origins = taxel_map.points_l.reshape(1, 1, 3, 3, 3)
    directions = taxel_map.normals_l.reshape(1, 1, 3, 3, 3)
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.01, max_distance_m=0.02)

    frame = source.frame_from_sphere(
        origins,
        directions,
        center_l=np.array([0.0, 0.0, 0.009], dtype=np.float32),
        radius_m=0.002,
    )

    pressure = frame.penetration_m[0, 0]
    np.testing.assert_allclose(pressure[1, 1], 0.003, atol=1.0e-8)
    assert pressure[1, 1] == np.max(pressure)
    assert pressure[0, 0] == 0.0
    assert frame.contact_mask[0, 0, 1, 1]


def test_geometry_normal_ray_source_box_and_triangle_mesh_agree():
    origins = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.01, max_distance_m=0.02)

    box_frame = source.frame_from_box(
        origins,
        directions,
        center_l=np.array([0.0, 0.0, 0.008], dtype=np.float32),
        half_extents_l=np.array([0.002, 0.002, 0.001], dtype=np.float32),
    )
    vertices = np.array(
        [
            [-0.002, -0.002, 0.007],
            [0.002, -0.002, 0.007],
            [0.002, 0.002, 0.007],
            [-0.002, 0.002, 0.007],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh_frame = source.frame_from_triangle_mesh(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
    )

    np.testing.assert_allclose(box_frame.penetration_m, 0.003, atol=1.0e-8)
    np.testing.assert_allclose(mesh_frame.penetration_m, box_frame.penetration_m, atol=1.0e-8)
    assert box_frame.contact_mask.tolist() == [[[[True]]]]
    assert mesh_frame.contact_mask.tolist() == [[[[True]]]]


def test_triangle_mesh_topology_diagnostics_distinguishes_closed_and_open_meshes():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    closed_faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    open_faces = np.array([[0, 1, 2]], dtype=np.int64)

    closed = triangle_mesh_topology_diagnostics(vertices, closed_faces)
    opened = triangle_mesh_topology_diagnostics(vertices, open_faces)

    assert closed["is_edge_watertight"] is True
    assert closed["boundary_edge_count"] == 0
    assert closed["nonmanifold_edge_count"] == 0
    assert closed["connected_component_count"] == 1
    assert opened["is_edge_watertight"] is False
    assert opened["boundary_edge_count"] == 3


def test_weld_duplicate_triangle_vertices_recovers_split_triangle_topology():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.arange(12, dtype=np.int64).reshape(4, 3)

    split = triangle_mesh_topology_diagnostics(vertices, triangles)
    welded_vertices, welded_triangles, summary = weld_duplicate_triangle_vertices(vertices, triangles)
    welded = triangle_mesh_topology_diagnostics(welded_vertices, welded_triangles)

    assert split["is_edge_watertight"] is False
    assert summary["original_vertex_count"] == 12
    assert summary["welded_vertex_count"] == 4
    assert welded["is_edge_watertight"] is True
    assert welded["connected_component_count"] == 1


def test_triangle_mesh_topology_diagnostics_reports_nonmanifold_edges():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )

    report = triangle_mesh_topology_diagnostics(vertices, faces)

    assert report["is_edge_watertight"] is False
    assert report["nonmanifold_edge_count"] == 1


def test_watertight_mesh_generator_outputs_closed_cylinder():
    module = _load_watertight_mesh_module()

    vertices, triangles = module.cylinder_mesh(
        center=(0.0, 0.0, 0.009),
        radius=0.002,
        height=0.002,
        segments=12,
    )
    report = triangle_mesh_topology_diagnostics(vertices, triangles)

    assert vertices.shape == (26, 3)
    assert triangles.shape == (48, 3)
    assert report["is_edge_watertight"] is True
    assert report["is_orientable_watertight"] is True
    assert report["boundary_edge_count"] == 0


def test_geometry_normal_ray_closed_mesh_inside_exit_filters_proximity():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.0],
            [0.001, -0.001, 0.0],
            [0.001, 0.001, 0.0],
            [-0.001, 0.001, 0.0],
            [-0.001, -0.001, 0.002],
            [0.001, -0.001, 0.002],
            [0.001, 0.001, 0.002],
            [-0.001, 0.001, 0.002],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    origins = np.array([[[[[0.0, 0.0, -0.001], [0.0, 0.0, 0.001]]]]], dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.01)

    frame = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
    )

    np.testing.assert_allclose(frame.penetration_m[0, 0, 0, 0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(frame.penetration_m[0, 0, 0, 1], 0.001, atol=1.0e-8)
    assert frame.contact_mask.tolist() == [[[[False, True]]]]


def test_geometry_normal_ray_closed_mesh_inside_exit_requires_bidirectional_inside():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.001],
            [0.001, -0.001, 0.001],
            [0.001, 0.001, 0.001],
            [-0.001, 0.001, 0.001],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    origins = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.01)

    frame = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
    )

    np.testing.assert_allclose(frame.penetration_m, 0.0, atol=1.0e-8)
    assert frame.contact_mask.tolist() == [[[[False]]]]


def test_geometry_normal_ray_closed_mesh_inside_exit_uses_nearest_boundary():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.0],
            [0.001, -0.001, 0.0],
            [0.001, 0.001, 0.0],
            [-0.001, 0.001, 0.0],
            [-0.001, -0.001, 0.010],
            [0.001, -0.001, 0.010],
            [0.001, 0.001, 0.010],
            [-0.001, 0.001, 0.010],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    origins = np.array([[[[[0.0, 0.0, 0.001]]]]], dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.02)

    frame = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
    )

    np.testing.assert_allclose(frame.penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    assert frame.contact_mask.tolist() == [[[[True]]]]


def test_geometry_normal_ray_closed_mesh_inside_can_subsample_taxel_area():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.0],
            [0.001, -0.001, 0.0],
            [0.001, 0.001, 0.0],
            [-0.001, 0.001, 0.0],
            [-0.001, -0.001, 0.002],
            [0.001, -0.001, 0.002],
            [0.001, 0.001, 0.002],
            [-0.001, 0.001, 0.002],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    origins = np.array([[[[[0.00125, 0.0, 0.001]]]]], dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.02)

    center_only = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
    )
    sampled = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=np.array([[0.0, 0.0, 0.0], [-0.0005, 0.0, 0.0]], dtype=np.float32),
        sample_aggregation="max",
    )

    np.testing.assert_allclose(center_only.penetration_m[0, 0, 0, 0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(sampled.penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    assert sampled.contact_mask.tolist() == [[[[True]]]]


def test_geometry_normal_ray_sampled_penetration_support_fraction_gate():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.0],
            [0.001, -0.001, 0.0],
            [0.001, 0.001, 0.0],
            [-0.001, 0.001, 0.0],
            [-0.001, -0.001, 0.002],
            [0.001, -0.001, 0.002],
            [0.001, 0.001, 0.002],
            [-0.001, 0.001, 0.002],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    origins = np.array([[[[[0.00125, 0.0, 0.001]]]]], dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    offsets = np.array(
        [[0.0, 0.0, 0.0], [-0.0005, 0.0, 0.0], [-0.00075, 0.0, 0.0]],
        dtype=np.float32,
    )
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.02)

    ungated_mean = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=offsets,
        sample_aggregation="mean",
    )
    gated_mean = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=offsets,
        sample_aggregation="mean",
        sample_min_support_fraction=0.75,
    )
    gated_max = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=offsets,
        sample_aggregation="max",
        sample_min_support_fraction=0.5,
    )

    np.testing.assert_allclose(ungated_mean.penetration_m[0, 0, 0, 0], 2.0e-3 / 3.0, atol=1.0e-8)
    np.testing.assert_allclose(gated_mean.penetration_m[0, 0, 0, 0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(gated_max.penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    assert ungated_mean.contact_mask.tolist() == [[[[True]]]]
    assert gated_mean.contact_mask.tolist() == [[[[False]]]]
    assert gated_max.contact_mask.tolist() == [[[[True]]]]


def test_geometry_normal_ray_sampled_penetration_stats_explain_boundary_support():
    vertices = np.array(
        [
            [-0.001, -0.001, 0.0],
            [0.001, -0.001, 0.0],
            [0.001, 0.001, 0.0],
            [-0.001, 0.001, 0.0],
            [-0.001, -0.001, 0.002],
            [0.001, -0.001, 0.002],
            [0.001, 0.001, 0.002],
            [-0.001, 0.001, 0.002],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    origins = np.array([[[[[0.00125, 0.0, 0.001]]]]], dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    offsets = np.array(
        [[0.0, 0.0, 0.0], [-0.0005, 0.0, 0.0], [-0.00075, 0.0, 0.0]],
        dtype=np.float32,
    )
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.0, max_distance_m=0.02)

    stats = source.closed_triangle_mesh_inside_sampled_penetration_stats(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=offsets,
        sample_aggregation="mean",
    )
    gated_frame = source.frame_from_closed_triangle_mesh_inside(
        origins,
        directions,
        vertices_l=vertices,
        triangles=faces,
        surface_sample_offsets_l=offsets,
        sample_aggregation="mean",
        sample_min_support_fraction=0.75,
    )

    np.testing.assert_allclose(stats.penetration_m[0, 0, 0, 0], 2.0e-3 / 3.0, atol=1.0e-8)
    np.testing.assert_allclose(stats.support_fraction[0, 0, 0, 0], 2.0 / 3.0, atol=1.0e-7)
    np.testing.assert_allclose(stats.active_count[0, 0, 0, 0], 2.0, atol=1.0e-8)
    np.testing.assert_allclose(stats.mean_penetration_m[0, 0, 0, 0], 2.0e-3 / 3.0, atol=1.0e-8)
    np.testing.assert_allclose(stats.positive_mean_penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    np.testing.assert_allclose(stats.max_penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    np.testing.assert_allclose(
        stats.sample_penetrations_m[0, 0, 0, 0],
        np.array([0.0, 0.001, 0.001], dtype=np.float32),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(stats.sample_offsets_l[0, 0, 0, 0], offsets, atol=1.0e-8)
    np.testing.assert_allclose(
        stats.sample_points_l_m[0, 0, 0, 0, :, 0],
        np.array([0.00125, 0.00075, 0.0005], dtype=np.float32),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(gated_frame.penetration_m[0, 0, 0, 0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(gated_frame.sample_support_fraction[0, 0, 0, 0], 2.0 / 3.0, atol=1.0e-7)
    np.testing.assert_allclose(gated_frame.sample_mean_penetration_m[0, 0, 0, 0], 2.0e-3 / 3.0, atol=1.0e-8)
    np.testing.assert_allclose(gated_frame.sample_positive_mean_penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    np.testing.assert_allclose(gated_frame.sample_max_penetration_m[0, 0, 0, 0], 0.001, atol=1.0e-8)
    np.testing.assert_allclose(gated_frame.sample_penetrations_m[0, 0, 0, 0], stats.sample_penetrations_m[0, 0, 0, 0])
    np.testing.assert_allclose(gated_frame.sample_offsets_l[0, 0, 0, 0], offsets, atol=1.0e-8)
    np.testing.assert_allclose(
        gated_frame.sample_points_l_m[0, 0, 0, 0],
        stats.sample_points_l_m[0, 0, 0, 0],
        atol=1.0e-8,
    )
    assert gated_frame.contact_mask.tolist() == [[[[False]]]]


def test_geometry_normal_ray_source_respects_max_distance_as_miss():
    origins = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1.0
    source = GeometryNormalRayPenetrationSource(rest_distance_m=0.01, max_distance_m=0.005)

    frame = source.frame_from_sphere(
        origins,
        directions,
        center_l=np.array([0.0, 0.0, 0.009], dtype=np.float32),
        radius_m=0.002,
    )

    np.testing.assert_allclose(frame.penetration_m, 0.0, atol=1.0e-8)
    assert frame.contact_mask.tolist() == [[[[False]]]]


def test_geometry_normal_ray_trace_uses_layout_without_tacmap_reference():
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=3,
        num_cols=3,
        point_distance=0.002,
        normal_axis=2,
    )
    trace = {
        "step": np.arange(2, dtype=np.int64),
        "pressure_taxel_points_l_m": taxel_map.points_l.reshape(1, 3, 3, 3),
        "pressure_taxel_normals_l": taxel_map.normals_l.reshape(1, 3, 3, 3),
    }

    out = apply_geometry_normal_ray_to_trace(
        trace,
        geometry={
            "kind": "sphere",
            "center_l_m": np.array([[0.0, 0.0, 0.012], [0.0, 0.0, 0.009]], dtype=np.float32),
            "radius_m": 0.002,
        },
        rest_distance_m=0.01,
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )

    pressure = out["geometry_normal_ray_penetration_m"]
    assert pressure.shape == (2, 1, 3, 3)
    np.testing.assert_allclose(pressure[0], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(pressure[1, 0, 1, 1], 0.003, atol=1.0e-8)
    assert "tacmap_raw_m" not in out


def test_write_geometry_normal_ray_trace_adds_independent_arrays(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=1,
        num_cols=1,
        point_distance=0.002,
        normal_axis=2,
    )
    trace_path = tmp_path / "layout.npz"
    out_path = tmp_path / "geometry_normal_ray.npz"
    np.savez_compressed(
        trace_path,
        pressure_taxel_points_l_m=taxel_map.points_l.reshape(1, 1, 1, 3),
        pressure_taxel_normals_l=taxel_map.normals_l.reshape(1, 1, 1, 3),
    )

    written, summary = write_geometry_normal_ray_trace(
        trace_path,
        out_path,
        geometry={
            "kind": "box",
            "center_l_m": np.array([0.0, 0.0, 0.008], dtype=np.float32),
            "half_extents_l_m": np.array([0.002, 0.002, 0.001], dtype=np.float32),
        },
        rest_distance_m=0.01,
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )

    assert written == out_path
    assert summary["geometry_kind"] == "box"
    with np.load(out_path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["geometry_normal_ray_penetration_m"], 0.003, atol=1.0e-8)
        assert data["geometry_normal_ray_contact_mask"][0, 0, 0, 0] == 1


def test_write_geometry_normal_ray_trace_reports_mesh_topology(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=1,
        num_cols=1,
        point_distance=0.002,
        normal_axis=2,
    )
    trace_path = tmp_path / "layout.npz"
    out_path = tmp_path / "geometry_mesh.npz"
    np.savez_compressed(
        trace_path,
        pressure_taxel_points_l_m=taxel_map.points_l.reshape(1, 1, 1, 3),
        pressure_taxel_normals_l=taxel_map.normals_l.reshape(1, 1, 1, 3),
    )
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    _written, summary = write_geometry_normal_ray_trace(
        trace_path,
        out_path,
        geometry={
            "kind": "mesh",
            "vertices_l_m": vertices,
            "triangles": faces,
        },
        rest_distance_m=0.01,
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )

    topology = summary["geometry_topology"]
    assert topology["is_edge_watertight"] is False
    assert topology["boundary_edge_count"] == 3


def test_write_geometry_normal_ray_trace_watertight_mesh_matches_l0_analytic(tmp_path):
    taxel_map = PressureTaxelMap.from_grid(
        link_name="finger_tip",
        num_rows=5,
        num_cols=5,
        point_distance=0.001,
        normal_axis=2,
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )
    trace = generate_l0_pressure_trace(
        taxel_map,
        AnalyticPresserSpec(kind="square", size_m=0.004),
        AnalyticPressTrajectory(steps=1, indentation_start_m=0.002, indentation_end_m=0.002),
    )
    trace_path, _metadata_path = trace.save(tmp_path, run_id="l0_square")
    out_path = tmp_path / "geometry_mesh.npz"
    vertices = np.array(
        [
            [-0.002, -0.002, 0.008],
            [0.002, -0.002, 0.008],
            [0.002, 0.002, 0.008],
            [-0.002, 0.002, 0.008],
            [-0.002, -0.002, 0.010],
            [0.002, -0.002, 0.010],
            [0.002, 0.002, 0.010],
            [-0.002, 0.002, 0.010],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )

    _written, summary = write_geometry_normal_ray_trace(
        trace_path,
        out_path,
        geometry={
            "kind": "mesh",
            "vertices_l_m": vertices,
            "triangles": faces,
        },
        rest_distance_m=0.01,
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )
    report = pressure_trace_report(
        out_path,
        pressure_key="geometry_normal_ray_pressure_norm",
        raw_pressure_key="geometry_normal_ray_pressure_raw_n",
        penetration_key="geometry_normal_ray_penetration_m",
        reference_key="analytic_depth_m",
        active_threshold=1.0e-6,
    )
    evaluation = evaluate_pressure_trace_report(
        report,
        reference_iou_threshold=1.0,
        centroid_error_threshold_px=0.0,
        bbox_error_threshold_px=0.0,
        depth_rmse_threshold_m=1.0e-9,
        onset_error_threshold_frames=0,
        offset_error_threshold_frames=0,
    )

    assert summary["geometry_topology"]["is_edge_watertight"] is True
    assert report["reference"]["active_mask_iou_min"] == 1.0
    assert evaluation["passed"] is True


def test_urdf_pressure_layout_source_exports_canonical_layouts(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_tip">
            <pressure_pad rows="2" cols="2" point_distance="0.001" normal_axis="2" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    source = UrdfPressureLayoutSource(urdf)
    layouts = source.layouts()
    maps = source.taxel_maps()

    assert len(layouts) == 1
    assert len(maps) == 1
    assert layouts[0].sensor_id == 0
    assert layouts[0].link_name == "finger_tip"
    assert layouts[0].image_shape == (2, 2)
    assert layouts[0].source == "urdf_pressure_layout"


def test_urdf_pressure_layout_source_respects_base_dir_for_npy_layouts(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    points = np.zeros((2, 2, 3), dtype=np.float32)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    np.save(assets / "points.npy", points)
    np.save(assets / "normals.npy", normals)
    urdf = tmp_path / "urdf" / "hand.urdf"
    urdf.parent.mkdir()
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_tip">
            <pressure_taxel_map points_npy="points.npy" normals_npy="normals.npy" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    source = UrdfPressureLayoutSource(urdf, base_dir=assets)
    maps = source.taxel_maps()
    layouts = source.layouts()

    assert maps[0].image_shape == (2, 2)
    assert layouts[0].link_name == "finger_tip"
    assert layouts[0].points_l_m.shape == (4, 3)


def test_urdf_pressure_layout_source_can_defer_missing_npy_file_checks(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="finger_tip">
            <pressure_taxel_map points_npy="missing_points.npy" normals_npy="missing_normals.npy" />
          </link>
        </robot>
        """,
        encoding="utf-8",
    )

    source = UrdfPressureLayoutSource(urdf, require_files=False)

    assert source.specs()[0].points_npy.name == "missing_points.npy"
    with pytest.raises(FileNotFoundError, match="missing_points.npy"):
        source.taxel_maps()


def test_urdf_pressure_layout_source_lists_pressure_touch_candidates(tmp_path):
    urdf = tmp_path / "hand.urdf"
    urdf.write_text(
        """
        <robot name="test_hand">
          <link name="right_indexmcp_roll_touch_link" />
          <link name="right_indexpip_roll_touch_link" />
          <link name="right_index_tip_touch_link" />
        </robot>
        """,
        encoding="utf-8",
    )

    source = UrdfPressureLayoutSource(urdf)

    assert source.touch_links() == [
        "right_indexmcp_roll_touch_link",
        "right_indexpip_roll_touch_link",
        "right_index_tip_touch_link",
    ]
    assert source.pressure_touch_links() == [
        "right_indexmcp_roll_touch_link",
        "right_indexpip_roll_touch_link",
    ]


def test_physx_contact_source_normalizes_sparse_records():
    taxel_map = _grid(PressureCalibration(stiffness=1.0, max_force=10.0))
    source = PhysxContactSource([taxel_map])
    events = source.events_from_records(
        [
            {
                "body_name": "finger_tip",
                "point_l": [0.0, 0.0, 0.0],
                "normal_l": [0.0, 0.0, 1.0],
                "force": 3.0,
            }
        ]
    )

    output = CalibratedPressureMapSensor([taxel_map], kernel_sigma=0.5).from_contact_events(events)

    assert events["finger_tip"].normal_forces.shape == (1,)
    assert float(events["finger_tip"].normal_forces[0]) == 3.0
    assert output.force_map.shape == (1, 5, 5)
    assert output.force_map[0, 2, 2] > 0.0


def test_physx_contact_source_projects_local_force_vector():
    taxel_map = _grid(PressureCalibration(stiffness=1.0, max_force=10.0))
    source = PhysxContactSource([taxel_map])
    events = source.events_from_records(
        [
            {
                "body_name": "finger_tip",
                "point_l": [0.0, 0.0, 0.0],
                "normal_l": [0.0, 0.0, 2.0],
                "contact_force_l": [1.0, 0.0, 4.0],
            }
        ]
    )

    assert float(events["finger_tip"].normal_forces[0]) == 4.0


def test_physx_contact_source_treats_vector_force_as_vector():
    taxel_map = _grid(PressureCalibration(stiffness=1.0, max_force=10.0))
    source = PhysxContactSource([taxel_map])
    events = source.events_from_records(
        [
            {
                "body_name": "finger_tip",
                "point_l": [0.0, 0.0, 0.0],
                "normal_l": [0.0, 0.0, 1.0],
                "force": [1.0, 0.0, 4.0],
            }
        ]
    )

    assert float(events["finger_tip"].normal_forces[0]) == 4.0


def test_physx_contact_source_treats_vector_force_with_world_normal_as_world():
    taxel_map = _grid(PressureCalibration(stiffness=1.0, max_force=10.0))
    source = PhysxContactSource([taxel_map])
    quat_x90 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)

    events = source.events_from_records(
        [
            {
                "body_name": "finger_tip",
                "point_l": [0.0, 0.0, 0.0],
                "contact_normal_w": [0.0, 0.0, 1.0],
                "force": [0.0, 0.0, 4.0],
            }
        ],
        link_poses_w={"finger_tip": (np.zeros(3, dtype=np.float32), quat_x90)},
    )

    assert float(events["finger_tip"].normal_forces[0]) == 4.0


def test_l1_deadband_fit_recovers_reference_onset():
    penetration = np.zeros((5, 1, 3, 3), dtype=np.float32)
    penetration[1:, 0, 1, 1] = np.array([0.0002, 0.0004, 0.0006, 0.0008], dtype=np.float32)
    reference = np.maximum(penetration - 0.0004, 0.0)
    trace = {
        "penetration_m": penetration,
        "pressure_norm": np.divide(
            penetration,
            np.maximum(np.max(penetration, axis=(-2, -1), keepdims=True), 1.0e-12),
        ),
        "pressure_raw_n": penetration,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_deadband_to_reference(
        trace,
        config=PressureDeadbandFitConfig(candidate_count=2),
        candidates_m=[0.0, 0.0002, 0.0004, 0.0006],
    )

    assert abs(result["best"]["deadband_m"] - 0.0004) < 1.0e-12
    assert result["best"]["reference_layer"] == "L1_model_reference"
    assert result["best"]["inferred_reference_layer"] == "L1_model_reference"
    assert result["best"]["reference_layer_override_conflict"] is False
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["best"]["onset_error_frames_by_sensor"] == [0]
    assert result["suggested_args"] == ["--penetration-deadband", "0.0004"]


def test_l1_deadband_fit_rejects_raw_sample_tensor_as_reference():
    penetration = np.zeros((2, 1, 2, 2), dtype=np.float32)
    trace = {
        "penetration_m": penetration,
        "pressure_norm": penetration,
        "pressure_raw_n": penetration,
        "geometry_normal_ray_sample_penetrations_m": np.zeros((2, 1, 2, 2, 3), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="raw per-sample tensors are diagnostic-only"):
        fit_pressure_deadband_to_reference(
            trace,
            config=PressureDeadbandFitConfig(
                reference_key="geometry_normal_ray_sample_penetrations_m",
                candidate_count=1,
            ),
        )


def test_l1_deadband_fit_respects_custom_reference_valid_mask():
    penetration = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration[1, 0, 0, 0] = 0.001
    penetration[1, 0, 1, 1] = 0.001
    valid_mask = np.array([[[1, 1], [1, 0]]], dtype=np.uint8)
    trace = {
        "penetration_m": penetration,
        "pressure_norm": penetration,
        "pressure_raw_n": penetration,
        "tacmap_raw_m": penetration,
        "custom_reference_valid_mask": valid_mask,
    }

    result = fit_pressure_deadband_to_reference(
        trace,
        config=PressureDeadbandFitConfig(
            candidate_count=1,
            max_deadband_m=0.0,
            reference_valid_mask_key="custom_reference_valid_mask",
        ),
    )

    assert result["best"]["reference_valid_mask_key"] == "custom_reference_valid_mask"
    assert result["best"]["alignment_contact_region_valid_fraction_min"] == 0.5
    assert result["best"]["alignment_invalid_contact_fraction_max"] == 0.5
    assert "reference.alignment_contact_region_valid_fraction_min" in result["best"]["failed_checks"]
    assert "reference.alignment_invalid_contact_fraction_max" in result["best"]["failed_checks"]


def test_l1_deadband_fit_can_search_depth_scale():
    penetration = np.zeros((5, 1, 3, 3), dtype=np.float32)
    penetration[1:, 0, 1, 1] = np.array([0.0004, 0.0008, 0.0012, 0.0016], dtype=np.float32)
    reference = np.maximum(0.5 * penetration - 0.0002, 0.0)
    trace = {
        "penetration_m": penetration,
        "pressure_norm": np.divide(
            penetration,
            np.maximum(np.max(penetration, axis=(-2, -1), keepdims=True), 1.0e-12),
        ),
        "pressure_raw_n": penetration,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_deadband_to_reference(
        trace,
        config=PressureDeadbandFitConfig(candidate_count=2),
        candidates_m=[0.0, 0.0002, 0.0004],
        scales=[0.25, 0.5, 1.0],
    )

    assert abs(result["best"]["scale"] - 0.5) < 1.0e-12
    assert abs(result["best"]["deadband_m"] - 0.0002) < 1.0e-12
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["suggested_depth_scale"] == "0.5"


def test_apply_pressure_deadband_to_trace_adds_fit_arrays():
    penetration = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration[1, 0, 0, 0] = 0.001
    trace = {"penetration_m": penetration, "pressure_norm": penetration, "pressure_raw_n": penetration}

    fitted = apply_pressure_deadband_to_trace(trace, deadband_m=0.00025, scale=0.5)

    assert "penetration_fit_m" in fitted
    assert "pressure_norm_fit" in fitted
    assert "pressure_raw_fit_n" in fitted
    np.testing.assert_allclose(fitted["penetration_fit_m"][1, 0, 0, 0], 0.00025, atol=1.0e-9)
    np.testing.assert_allclose(fitted["pressure_norm_fit"].max(), 1.0, atol=1.0e-6)


def test_apply_pressure_mask_depth_to_trace_splits_active_mask_from_depth():
    penetration = np.zeros((2, 1, 2, 2), dtype=np.float32)
    penetration[1, 0, 0, 0] = 0.0003
    penetration[1, 0, 0, 1] = 0.001
    trace = {"penetration_m": penetration, "pressure_norm": penetration, "pressure_raw_n": penetration}

    fitted = apply_pressure_mask_depth_to_trace(
        trace,
        mask_deadband_m=0.0002,
        depth_deadband_m=0.0008,
        scale=1.0,
        active_floor_m=1.0e-6,
    )

    assert "penetration_mask_depth_fit_m" in fitted
    assert "pressure_norm_mask_depth_fit" in fitted
    assert "pressure_raw_mask_depth_fit_n" in fitted
    assert fitted["pressure_norm_mask_depth_fit"][1, 0, 0, 0] > 0.0
    np.testing.assert_allclose(fitted["penetration_mask_depth_fit_m"][1, 0, 0, 0], 1.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(fitted["penetration_mask_depth_fit_m"][1, 0, 0, 1], 0.0002, atol=1.0e-9)


def test_l1_mask_depth_fit_can_keep_edge_mask_while_fitting_depth():
    penetration = np.zeros((3, 1, 2, 2), dtype=np.float32)
    reference = np.zeros_like(penetration)
    penetration[1, 0, 0, 0] = 0.0003
    penetration[1, 0, 0, 1] = 0.001
    penetration[1, 0, 1, 1] = 0.0001
    penetration[2, 0, 0, 0] = 0.0004
    penetration[2, 0, 0, 1] = 0.0012
    penetration[2, 0, 1, 1] = 0.0001
    reference[1, 0, 0, 0] = 1.0e-6
    reference[1, 0, 0, 1] = 0.0002
    reference[2, 0, 0, 0] = 1.0e-6
    reference[2, 0, 0, 1] = 0.0004
    trace = {
        "penetration_m": penetration,
        "pressure_norm": penetration,
        "pressure_raw_n": penetration,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_mask_depth_to_reference(
        trace,
        mask_deadbands_m=[0.0002, 0.0008],
        depth_deadbands_m=[0.0008],
        scales=[1.0],
        active_floor_m=[1.0e-6],
    )

    assert 0.0 < result["best"]["mask_deadband_m"] <= 0.0002
    assert abs(result["best"]["depth_deadband_m"] - 0.0008) < 1.0e-12
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["best"]["onset_error_frames_by_sensor"] == [0]


def test_apply_pressure_sample_support_to_trace_weights_depth_by_support():
    support = np.zeros((2, 1, 2, 2), dtype=np.float32)
    depth = np.zeros_like(support)
    support[1, 0, 0, 0] = 0.5
    depth[1, 0, 0, 0] = 0.002
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": depth,
        "penetration_m": depth,
        "pressure_norm": depth,
        "pressure_raw_n": depth,
    }

    fitted = apply_pressure_sample_support_to_trace(
        trace,
        support_threshold=0.25,
        mask_deadband_m=0.0,
        depth_deadband_m=0.0,
        scale=1.0,
        support_power=1.0,
    )

    assert "penetration_sample_support_fit_m" in fitted
    assert "pressure_norm_sample_support_fit" in fitted
    assert "pressure_raw_sample_support_fit_n" in fitted
    np.testing.assert_allclose(fitted["penetration_sample_support_fit_m"][1, 0, 0, 0], 0.001, atol=1.0e-9)
    np.testing.assert_allclose(fitted["pressure_norm_sample_support_fit"].max(), 1.0, atol=1.0e-6)


def test_l1_sample_support_fit_can_filter_low_support_false_positive():
    support = np.zeros((3, 1, 2, 2), dtype=np.float32)
    depth = np.zeros_like(support)
    reference = np.zeros_like(support)
    support[1:, 0, 0, 0] = 0.25
    depth[1:, 0, 0, 0] = 0.001
    support[1:, 0, 0, 1] = 0.75
    depth[1:, 0, 0, 1] = 0.001
    reference[1:, 0, 0, 1] = 0.001
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": depth,
        "penetration_m": depth,
        "pressure_norm": depth,
        "pressure_raw_n": depth,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_sample_support_to_reference(
        trace,
        config=PressureSampleSupportFitConfig(candidate_count=2),
        support_thresholds=[0.0, 0.5],
        mask_deadbands_m=[0.0],
        depth_deadbands_m=[0.0],
        scales=[1.0],
        support_powers=[0.0],
        depth_source_keys=["geometry_normal_ray_sample_mean_penetration_m"],
        active_floor_m=[0.0],
    )

    assert result["best"]["support_threshold"] == 0.5
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["best"]["onset_error_frames_by_sensor"] == [0]


def test_apply_pressure_area_fraction_to_trace_integrates_support_and_max_depth():
    support = np.zeros((2, 1, 2, 2), dtype=np.float32)
    mean = np.zeros_like(support)
    positive = np.zeros_like(support)
    max_depth = np.zeros_like(support)
    support[1, 0, 0, 0] = 0.5
    mean[1, 0, 0, 0] = 0.001
    positive[1, 0, 0, 0] = 0.002
    max_depth[1, 0, 0, 0] = 0.003
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": mean,
        "geometry_normal_ray_sample_positive_mean_penetration_m": positive,
        "geometry_normal_ray_sample_max_penetration_m": max_depth,
        "geometry_normal_ray_penetration_m": positive,
        "penetration_m": mean,
        "pressure_norm": mean,
        "pressure_raw_n": mean,
    }

    fitted = apply_pressure_area_fraction_to_trace(
        trace,
        area_mode="support_max",
        support_threshold=0.25,
        mask_deadband_m=0.0,
        depth_deadband_m=0.0,
        scale=1.0,
    )

    assert "penetration_area_fraction_fit_m" in fitted
    assert "pressure_norm_area_fraction_fit" in fitted
    assert "pressure_raw_area_fraction_fit_n" in fitted
    np.testing.assert_allclose(fitted["penetration_area_fraction_fit_m"][1, 0, 0, 0], 0.0015, atol=1.0e-9)
    np.testing.assert_allclose(fitted["pressure_norm_area_fraction_fit"].max(), 1.0, atol=1.0e-6)


def test_l1_area_fraction_fit_can_choose_depth_integration_mode():
    support = np.zeros((3, 1, 2, 2), dtype=np.float32)
    mean = np.zeros_like(support)
    positive = np.zeros_like(support)
    max_depth = np.zeros_like(support)
    reference = np.zeros_like(support)
    support[1:, 0, 0, 0] = 0.5
    mean[1:, 0, 0, 0] = 0.001
    positive[1:, 0, 0, 0] = 0.002
    max_depth[1:, 0, 0, 0] = 0.003
    reference[1:, 0, 0, 0] = 0.0015
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": mean,
        "geometry_normal_ray_sample_positive_mean_penetration_m": positive,
        "geometry_normal_ray_sample_max_penetration_m": max_depth,
        "geometry_normal_ray_penetration_m": positive,
        "penetration_m": mean,
        "pressure_norm": mean,
        "pressure_raw_n": mean,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_area_fraction_to_reference(
        trace,
        config=PressureAreaFractionFitConfig(candidate_count=2),
        area_modes=["mean", "support_max"],
        blend_weights=[0.0],
        support_thresholds=[0.0],
        mask_deadbands_m=[0.0],
        depth_deadbands_m=[0.0],
        scales=[1.0],
        active_floor_m=[0.0],
    )

    assert result["best"]["area_mode"] == "support_max"
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["best"]["depth_rmse_m_max"] == 0.0


def test_apply_pressure_spatial_footprint_to_trace_erodes_early_and_dilates_late():
    support = np.ones((3, 1, 5, 5), dtype=np.float32)
    depth = np.zeros_like(support)
    depth[1, 0, 1:4, 1:4] = 0.001
    depth[2, 0, 2, 2] = 0.003
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": depth,
        "penetration_m": depth,
        "pressure_norm": depth,
        "pressure_raw_n": depth,
    }

    fitted = apply_pressure_spatial_footprint_to_trace(
        trace,
        support_threshold=0.0,
        mask_deadband_m=0.0,
        depth_deadband_m=0.0,
        scale=1.0,
        support_power=0.0,
        transition_depth_m=0.002,
        early_op="erode",
        early_iterations=1,
        late_op="dilate",
        late_iterations=1,
    )

    penetration = fitted["penetration_spatial_footprint_fit_m"]
    assert np.count_nonzero(penetration[1, 0] > 0.0) == 1
    assert penetration[1, 0, 2, 2] > 0.0
    assert np.count_nonzero(penetration[2, 0] > 0.0) == 9
    np.testing.assert_allclose(penetration[2, 0, 1, 1], 0.003, atol=1.0e-9)


def test_l1_spatial_footprint_fit_can_select_erode_then_dilate():
    support = np.ones((3, 1, 5, 5), dtype=np.float32)
    depth = np.zeros_like(support)
    reference = np.zeros_like(support)
    depth[1, 0, 1:4, 1:4] = 0.001
    reference[1, 0, 2, 2] = 0.001
    depth[2, 0, 2, 2] = 0.003
    reference[2, 0, 1:4, 1:4] = 0.003
    trace = {
        "geometry_normal_ray_sample_support_fraction": support,
        "geometry_normal_ray_sample_mean_penetration_m": depth,
        "penetration_m": depth,
        "pressure_norm": depth,
        "pressure_raw_n": depth,
        "tacmap_raw_m": reference,
    }

    result = fit_pressure_spatial_footprint_to_reference(
        trace,
        config=PressureSpatialFootprintFitConfig(candidate_count=2),
        support_thresholds=[0.0],
        mask_deadbands_m=[0.0],
        depth_deadbands_m=[0.0],
        scales=[1.0],
        support_powers=[0.0],
        depth_source_keys=["geometry_normal_ray_sample_mean_penetration_m"],
        active_floor_m=[0.0],
        transition_depths_m=[0.002],
        early_ops=["none", "erode"],
        late_ops=["none", "dilate"],
        early_iterations=[0, 1],
        late_iterations=[0, 1],
    )

    assert result["best"]["early_op"] == "erode"
    assert result["best"]["late_op"] == "dilate"
    assert result["best"]["active_mask_iou_min"] == 1.0
    assert result["best"]["depth_rmse_m_max"] == 0.0


def test_pressure_trace_report_prefers_requested_raw_pressure_key_for_total_force():
    penetration = np.zeros((3, 1, 2, 2), dtype=np.float32)
    pressure_fit = np.zeros_like(penetration)
    penetration[:, 0, 0, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    pressure_fit[:, 0, 0, 0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    stale_total = np.array([[2.0], [1.0], [0.0]], dtype=np.float32)

    report = pressure_trace_report(
        {
            "penetration_fit_m": penetration,
            "pressure_norm_fit": pressure_fit,
            "pressure_raw_fit_n": pressure_fit,
            "total_force_n": stale_total,
        },
        pressure_key="pressure_norm_fit",
        raw_pressure_key="pressure_raw_fit_n",
        penetration_key="penetration_fit_m",
    )

    assert report["force_depth_spearman_min"] == 1.0


def test_geometry_alignment_resamples_reference_to_pressure_taxels():
    pressure_points = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    reference_points = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0]],
                [[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    reference = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)

    aligned, summary = align_reference_to_pressure_taxels(
        {
            "tacmap_raw_m": reference,
            "pressure_taxel_points_l_m": pressure_points,
            "tacmap_grid_points_l_m": reference_points,
            "pressure_taxel_layout_valid": np.array([1], dtype=np.uint8),
            "tacmap_grid_layout_valid": np.array([1], dtype=np.uint8),
        }
    )

    np.testing.assert_allclose(aligned["tacmap_raw_aligned_m"][0, 0], np.array([[0, 2], [6, 8]], dtype=np.float32))
    assert summary["aligned_shape"] == [1, 1, 2, 2]
    assert summary["sensors"][0]["valid_taxel_count"] == 4


def test_aligned_reference_points_for_pressure_taxels_uses_same_alignment_indices():
    reference_points = np.array(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            ]
        ],
        dtype=np.float32,
    )
    fallback_points = np.full((1, 1, 3, 3), 9.0, dtype=np.float32)
    source_index = np.array([[[0, 3, -1]]], dtype=np.int64)
    valid_mask = np.array([[[1, 1, 0]]], dtype=np.uint8)

    aligned_points, aligned_valid = aligned_reference_points_for_pressure_taxels(
        reference_points,
        source_index,
        valid_mask,
        fallback_points=fallback_points,
    )

    np.testing.assert_allclose(aligned_points[0, 0, 0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(aligned_points[0, 0, 1], np.array([0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(aligned_points[0, 0, 2], np.array([9.0, 9.0, 9.0], dtype=np.float32))
    np.testing.assert_array_equal(aligned_valid, np.array([[[1, 1, 0]]], dtype=np.uint8))


def test_precomputed_geometry_alignment_matches_offline_alignment():
    pressure_points = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=np.float32)
    reference_points = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=np.float32)
    reference = np.array([[[[4.0, 9.0]]]], dtype=np.float32)

    source_index, valid_mask, _distance, summary = build_reference_to_pressure_alignment(
        pressure_points,
        reference_points,
        distance_mode="local3d",
    )
    aligned = apply_reference_alignment_to_values(reference, source_index, valid_mask)
    offline, _ = align_reference_to_pressure_taxels(
        {
            "tacmap_raw_m": reference,
            "pressure_taxel_points_l_m": pressure_points,
            "tacmap_grid_points_l_m": reference_points,
        },
        distance_mode="local3d",
    )

    np.testing.assert_allclose(aligned, offline["tacmap_raw_aligned_m"])
    np.testing.assert_allclose(aligned[0, 0], np.array([[4.0, 9.0]], dtype=np.float32))
    assert summary["sensors"][0]["valid_taxel_count"] == 2


def test_geometry_alignment_plane_mode_ignores_ray_axis_offset():
    pressure_points = np.array([[[[0.02, 1.0, 0.0]]]], dtype=np.float32)
    reference_points = np.array([[[[-0.01, 0.0, 0.0], [-0.01, 1.0, 0.0]]]], dtype=np.float32)
    reference = np.array([[[[3.0, 7.0]]]], dtype=np.float32)
    axes = np.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32)

    aligned, summary = align_reference_to_pressure_taxels(
        {
            "tacmap_raw_m": reference,
            "pressure_taxel_points_l_m": pressure_points,
            "tacmap_grid_points_l_m": reference_points,
            "tacmap_grid_axes_l": axes,
        },
        max_distance_m=0.001,
    )

    np.testing.assert_allclose(aligned["tacmap_raw_aligned_m"], np.array([[[[7.0]]]], dtype=np.float32))
    assert summary["distance_mode"] == "plane"
    assert summary["sensors"][0]["nearest_distance_m_max"] == 0.0


def test_write_geometry_aligned_trace_adds_alignment_arrays(tmp_path):
    trace_path = tmp_path / "trace.npz"
    out_path = tmp_path / "aligned.npz"
    np.savez_compressed(
        trace_path,
        tacmap_raw_m=np.array([[[[1.0, 2.0]]]], dtype=np.float32),
        pressure_taxel_points_l_m=np.array([[[[1.0, 0.0, 0.0]]]], dtype=np.float32),
        tacmap_grid_points_l_m=np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=np.float32),
        pressure_taxel_layout_valid=np.array([1], dtype=np.uint8),
        tacmap_grid_layout_valid=np.array([1], dtype=np.uint8),
    )

    written, summary = write_geometry_aligned_trace(trace_path, out_path)

    assert written == out_path
    assert summary["valid_sensor_indices"] == [0]
    with np.load(out_path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["tacmap_raw_aligned_m"], np.array([[[[2.0]]]], dtype=np.float32))
        assert data["tacmap_raw_aligned_m_valid_mask"].shape == (1, 1, 1)
        assert data["tacmap_raw_aligned_m_source_index"][0, 0, 0] == 1


def test_normal_ray_reference_trace_uses_deformation_as_penetration():
    deformation = np.zeros((3, 1, 2, 2), dtype=np.float32)
    deformation[1, 0, 0, 1] = 0.001
    deformation[2, 0, 1, 1] = 0.002
    trace = {"tacmap_raw_aligned_m": deformation}

    out = apply_normal_ray_reference_to_trace(
        trace,
        deformation_key="tacmap_raw_aligned_m",
        output_prefix="normal_ray",
        calibration=PressureCalibration(stiffness=1.0, max_force=1.0),
    )

    np.testing.assert_allclose(out["normal_ray_penetration_m"], deformation)
    np.testing.assert_allclose(out["normal_ray_signed_distance_m"], -deformation)
    assert out["normal_ray_pressure_norm"].shape == deformation.shape

    report = pressure_trace_report(
        out,
        pressure_key="normal_ray_pressure_norm",
        raw_pressure_key="normal_ray_pressure_raw_n",
        penetration_key="normal_ray_penetration_m",
        reference_key="tacmap_raw_aligned_m",
    )
    evaluation = evaluate_pressure_trace_report(
        report,
        reference_iou_threshold=1.0,
        centroid_error_threshold_px=0.0,
        bbox_error_threshold_px=0.0,
        depth_rmse_threshold_m=0.0,
    )

    assert report["reference"]["active_mask_iou_min"] == 1.0
    assert report["reference"]["reference_layer"] == "L1_model_reference"
    assert report["reference"]["depth_rmse_m_max"] == 0.0
    assert evaluation["passed"] is True


def test_write_normal_ray_reference_trace_adds_model_reference_arrays(tmp_path):
    trace_path = tmp_path / "aligned.npz"
    out_path = tmp_path / "normal_ray.npz"
    np.savez_compressed(
        trace_path,
        tacmap_raw_aligned_m=np.array([[[[0.0, 0.001]]]], dtype=np.float32),
    )

    written, summary = write_normal_ray_reference_trace(
        trace_path,
        out_path,
        deformation_key="tacmap_raw_aligned_m",
    )

    assert written == out_path
    assert summary["penetration_shape"] == [1, 1, 1, 2]
    with np.load(out_path, allow_pickle=False) as data:
        np.testing.assert_allclose(
            data["normal_ray_penetration_m"],
            np.array([[[[0.0, 0.001]]]], dtype=np.float32),
        )
        assert data["normal_ray_contact_mask"][0, 0, 0, 0] == 0
        assert data["normal_ray_contact_mask"][0, 0, 0, 1] == 1


def test_dv2_rl_pressure_layout_has_expected_runtime_order_and_285_taxels():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    layout_path = (
        repo_root
        / "assets"
        / "revo21_right_touch"
        / "pressure_taxels"
        / "dv2"
        / "dv2_pressure_taxel_layout.urdf"
    )
    expected_link_order = (
        "right_midmcp_roll_touch_link",
        "right_midpip_roll_touch_link",
        "right_indexmcp_roll_touch_link",
        "right_indexpip_roll_touch_link",
        "right_ringmcp_roll_touch_link",
        "right_ringpip_roll_touch_link",
        "right_pinkymcp_roll_touch_link",
        "right_pinkypip_roll_touch_link",
        "right_thumbmcp_roll_touch_link",
        "right_thumbpip_roll_touch_link",
        "right_hand_rubber_link",
    )
    expected_counts = (39, 10, 39, 10, 39, 10, 39, 10, 43, 10, 36)

    config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    constant_names = {
        "TIANJI_RL_FINGER_ORDER",
        "TIANJI_PRESSURE_PAD_FINGER_STEMS",
        "TIANJI_PRESSURE_PAD_SEGMENTS",
        "TIANJI_PRESSURE_PAD_LINK_ORDER",
        "TIANJI_PRESSURE_SENSOR_NAMES",
    }
    constant_nodes = [
        node
        for node in config_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in constant_names for target in node.targets)
    ]
    config_namespace = {}
    exec(compile(ast.Module(body=constant_nodes, type_ignores=[]), str(config_path), "exec"), config_namespace)
    link_order = config_namespace["TIANJI_PRESSURE_PAD_LINK_ORDER"]

    specs = {spec.link_name: spec for spec in load_pressure_pad_specs_from_urdf(layout_path)}
    maps = [specs[link_name].to_taxel_map() for link_name in link_order]

    assert link_order == expected_link_order
    assert len(config_namespace["TIANJI_PRESSURE_SENSOR_NAMES"]) == len(expected_link_order)
    assert tuple(taxel_map.num_taxels for taxel_map in maps) == expected_counts
    assert sum(taxel_map.num_taxels for taxel_map in maps) == 285
    assert all(taxel_map.image_shape == (1, count) for taxel_map, count in zip(maps, expected_counts, strict=True))
    assert all(np.isfinite(taxel_map.points_l).all() for taxel_map in maps)
    assert all(np.isfinite(taxel_map.normals_l).all() for taxel_map in maps)


def test_ours_observation_selection_gates_sensor_creation_and_component_calls():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Revo3MixinCfg")
    post_init = next(node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    source = ast.unparse(post_init)

    assert "ours_pressure_enabled = 'pressure' in selected_ours_terms" in source
    assert "ours_hydroshear_enabled = 'hydroshear' in selected_ours_terms" in source
    assert "ours_taxim_rgb_enabled = 'taxim_rgb' in selected_ours_terms" in source
    assert "ours_tacmap_needed = 'tacmap_policy' in selected_ours_terms or ours_hydroshear_enabled or ours_taxim_rgb_enabled" in source
    assert "if tactile_implementation == 'ours' and ours_pressure_enabled:" in source
    assert "if tactile_implementation == 'ours' and (not ours_tacmap_needed):\n            continue" in source
    assert "if tactile_implementation == 'ours' and ours_hydroshear_enabled:" in source
    assert "self.events.zz_ours_tactile_cache_reset" in source
    assert "func=mdp.invalidate_ours_tactile_cache_on_reset" in source
    assert "self.observations.proprio.rl_ours_taxim_rgb" in source
    assert "func=mdp.ours_rl_tacmap_resnet_obs" in source
    assert "func=mdp.ours_rl_taxim_resnet_obs" in source
    assert "'tacmap_policy_full_depth_image': True" in source
    assert "'tactile_resnet_output_dim': TIANJI_TACTILE_RESNET_OUTPUT_DIM" in source
    assert "'taxim_rgb_policy_rows'" not in source
    assert "'taxim_rgb_policy_cols'" not in source
    assert "release_cache" not in source


def test_rl_surface_visual_points_offset_only_the_rendered_copy():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "surface_visual_points"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    points = np.zeros((2, 3), dtype=np.float32)
    normals = np.array([[2.0, 0.0, 0.0], [0.0, -3.0, 0.0]], dtype=np.float32)
    rendered = namespace["surface_visual_points"](points, normals, 0.0005)

    np.testing.assert_array_equal(points, np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_allclose(rendered, [[0.0005, 0.0, 0.0], [0.0, -0.0005, 0.0]])


def test_rl_calibrated_marker_cache_uses_all_vitai_layouts_and_quality_colors():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calibrated_marker_local_cache"
    )
    namespace = {
        "np": np,
        "FINGER_CHOICES": ("middle", "index", "ring", "pinky", "thumb"),
        "TIANJI_TACMAP_LINK_ORDER": (
            "right_middip_roll_rubber_link",
            "right_indexdip_roll_rubber_link",
            "right_ringdip_roll_rubber_link",
            "right_pinkydip_roll_rubber_link",
            "right_thumbdip_roll_rubber_link",
        ),
        "VITAI_MARKER_LAYOUT": (
            repo_root
            / "assets"
            / "revo21_right_touch"
            / "marker_positions"
            / "vitai_4fingers"
            / "marker_positions.npz"
        ),
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    cache = namespace["calibrated_marker_local_cache"]()
    colors = np.concatenate([entry[3] for entry in cache], axis=0)

    assert len(cache) == 5
    assert [entry[1].shape for entry in cache] == [
        (86, 3),
        (86, 3),
        (86, 3),
        (85, 3),
        (87, 3),
    ]
    assert np.count_nonzero(np.all(np.isclose(colors, (0.0, 0.2, 1.0)), axis=1)) == 430
    assert np.count_nonzero(np.all(np.isclose(colors, (1.0, 0.0, 1.0)), axis=1)) == 0


def test_rl_hydroshear_calibrated_marker_loader_includes_thumb():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_hydroshear_calibrated_marker_world"
    )
    source = ast.unparse(function_node)

    assert "RL_FINGER_ORDER[:sensor_count]" in source
    assert "min(sensor_count, 4)" not in source


def test_calibrated_marker_visualizers_keep_centers_at_surface_positions():
    repo_root = Path(__file__).resolve().parents[1]
    rl_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    rl_tree = ast.parse(rl_path.read_text(encoding="utf-8"))
    rl_function = next(
        node
        for node in rl_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "calibrated_marker_world_arrays"
    )
    integrated_text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    integrated_block = integrated_text[
        integrated_text.index("class TacMapFingerMarkerPointViz")
        : integrated_text.index("class TacMapFingerMarkerNormalViz")
    ]

    assert "surface_visual_points" not in ast.unparse(rl_function)
    assert "marker_points_l +=" not in integrated_block


def test_rl_pressure_sampling_applies_one_mm_outward_offset():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    config_text = config_path.read_text(encoding="utf-8")
    visualizer_tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    visualizer_functions = {
        node.name: ast.unparse(node)
        for node in visualizer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"pressure_pad_target", "pressure_pad_taxel_world_arrays"}
    }

    assert "TIANJI_PRESSURE_PAD_YELLOW_TAXEL_NORMAL_OFFSET_M" not in config_text
    assert "points_l = points_l + normals_l * 0.001" in config_text
    assert "TIANJI_PRESSURE_PAD_YELLOW_TAXEL_NORMAL_OFFSET_M" not in visualizer_functions["pressure_pad_target"]
    assert "visual_offset_m" in visualizer_functions["pressure_pad_taxel_world_arrays"]
    assert "surface_visual_points" in visualizer_functions["pressure_pad_taxel_world_arrays"]


def test_rl_pressure_heatmap_preserves_palm_taxel_geometry():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pressure_layout_pixel_centers"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    layout_path = (
        repo_root
        / "assets"
        / "revo21_right_touch"
        / "pressure_taxels"
        / "dv2"
        / "dv2_pressure_taxel_layout.urdf"
    )
    specs = {spec.link_name: spec for spec in load_pressure_pad_specs_from_urdf(layout_path)}
    palm_spec = specs["right_hand_rubber_link"]
    points_l = np.asarray(palm_spec.to_taxel_map().points_l)
    centers, _height, _width = namespace["pressure_layout_pixel_centers"](points_l, spacing_px=20)

    pixel_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(pixel_distance, np.inf)
    assert int(np.argmin(pixel_distance[1])) == 34  # Palm IDs 02 and 35 are physical nearest-neighbors.

    thumb_points_l = np.asarray(specs["right_thumbmcp_roll_touch_link"].to_taxel_map().points_l)
    _centers, height, width = namespace["pressure_layout_pixel_centers"](thumb_points_l, spacing_px=20)
    assert height * width * 3 < 100 * 2**20


def test_pressure_display_diffusion_uses_physical_distance_and_conserves_total():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"build_pressure_display_diffusion_kernel", "diffuse_pressure_display_values"}
    }
    namespace = {"np": np}
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), str(visualizer_path), "exec"), namespace)

    points_l = np.array(
        [
            [0.0, 0.000, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.020, 0.0],
        ],
        dtype=np.float32,
    )
    normals_l = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
    kernel = namespace["build_pressure_display_diffusion_kernel"](
        points_l,
        normals_l,
        sigma_m=0.002,
        radius_sigma=3.0,
        normal_power=1.0,
    )
    raw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    display = namespace["diffuse_pressure_display_values"](raw, kernel, blend=1.0)

    np.testing.assert_allclose(np.sum(kernel, axis=0), np.ones(3), atol=1.0e-6)
    np.testing.assert_allclose(np.sum(display), np.sum(raw), atol=1.0e-6)
    assert display[0] > display[1] > 0.0
    assert display[2] == 0.0


def test_pressure_display_diffusion_respects_normals_and_does_not_mutate_input():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"build_pressure_display_diffusion_kernel", "diffuse_pressure_display_values"}
    }
    namespace = {"np": np}
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), str(visualizer_path), "exec"), namespace)

    points_l = np.array([[0.0, 0.000, 0.0], [0.0, 0.001, 0.0]], dtype=np.float32)
    normals_l = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    kernel = namespace["build_pressure_display_diffusion_kernel"](
        points_l,
        normals_l,
        sigma_m=0.002,
        radius_sigma=3.0,
        normal_power=1.0,
    )
    raw = np.array([1.0, 0.0], dtype=np.float32)
    before = raw.copy()
    display = namespace["diffuse_pressure_display_values"](raw, kernel, blend=1.0)

    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(kernel, np.eye(2), atol=1.0e-6)
    np.testing.assert_allclose(display, raw, atol=1.0e-6)
    for function_node in functions.values():
        assert "torch" not in {node.id for node in ast.walk(function_node) if isinstance(node, ast.Name)}


def test_pressure_display_and_rl_diffusion_are_connected_to_separate_runtime_paths():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    visualizer_text = visualizer_path.read_text(encoding="utf-8")
    assert "--pressure-display-diffusion" in visualizer_text

    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    config_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    observations_text = observations_path.read_text(encoding="utf-8")
    config_text = config_path.read_text(encoding="utf-8")
    assert "build_pressure_rl_diffusion_kernel" in observations_text
    assert "diffuse_pressure_rl_values" in observations_text
    assert '"pressure_diffusion_enabled": TIANJI_RL_PRESSURE_DIFFUSION_ENABLED' in config_text
    assert "TIANJI_RL_PRESSURE_DIFFUSION_ENABLED = True" in config_text

    non_diffusion_runtime_paths = (
        repo_root / "scripts" / "rsl_rl" / "train.py",
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "force_map"
        / "warp_sdf_tactile_sensor.py",
    )
    for path in non_diffusion_runtime_paths:
        text = path.read_text(encoding="utf-8")
        assert "pressure_display_diffusion" not in text
        assert "build_pressure_display_diffusion_kernel" not in text
        assert "build_pressure_rl_diffusion_kernel" not in text


def test_hydroshear_roi_wireframe_draws_near_far_and_side_boundaries():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    visualizer_text = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(visualizer_text)
    roi_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_hydroshear_roi_wireframe_segments"
    )
    module = ast.Module(body=[roi_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, filename=str(visualizer_path), mode="exec"), namespace)
    build_segments = namespace["build_hydroshear_roi_wireframe_segments"]

    points = np.zeros((2, 2, 3), dtype=np.float32)
    points[..., 0] = 1.0
    points[0, 0, 1:] = (0.0, 0.0)
    points[0, 1, 1:] = (1.0, 0.0)
    points[1, 0, 1:] = (0.0, 1.0)
    points[1, 1, 1:] = (1.0, 1.0)
    segments = build_segments(
        points,
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        surface_margin_m=0.1,
        invalid_depth_ratio=0.5,
        grid_stride=1,
    )

    assert segments.ndim == 3
    assert segments.shape[1:] == (2, 3)
    assert segments.shape[0] > 0
    assert np.isfinite(segments).all()
    assert float(np.min(segments[..., 0])) == pytest.approx(0.0)
    assert float(np.max(segments[..., 0])) == pytest.approx(1.1)
    assert np.any(
        np.isclose(segments[:, 0, 0], 0.0) & np.isclose(segments[:, 1, 0], 1.1)
    )


def test_hydroshear_roi_wireframe_preserves_non_affine_ray_start_range():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    visualizer_text = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(visualizer_text)
    roi_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_hydroshear_roi_wireframe_segments"
    )
    module = ast.Module(body=[roi_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, filename=str(visualizer_path), mode="exec"), namespace)
    build_segments = namespace["build_hydroshear_roi_wireframe_segments"]

    starts = np.zeros((2, 3, 3), dtype=np.float32)
    starts[0, :, 1] = (-2.0, 0.0, 2.0)
    starts[1, :, 1] = (-1.0, 0.0, 1.0)
    starts[1, :, 2] = 1.0
    raw = np.ones((2, 3), dtype=np.float32)
    ray = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    points = starts + raw[..., None] * ray

    segments = build_segments(
        points,
        raw,
        np.ones((2, 3), dtype=bool),
        ray,
        surface_margin_m=0.1,
        invalid_depth_ratio=0.5,
        grid_stride=1,
    )

    near_points = segments[np.isclose(segments[..., 0], 0.0)]
    assert float(np.min(near_points[..., 1])) == pytest.approx(-2.0)
    assert float(np.max(near_points[..., 1])) == pytest.approx(2.0)


def test_hydroshear_roi_wireframe_expands_half_cell_and_repeats_edge_depth():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    roi_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_hydroshear_roi_wireframe_segments"
    )
    module = ast.Module(body=[roi_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np}
    exec(compile(module, filename=str(visualizer_path), mode="exec"), namespace)
    build_segments = namespace["build_hydroshear_roi_wireframe_segments"]

    starts = np.zeros((2, 3, 3), dtype=np.float32)
    starts[:, :, 1] = (-2.0, 0.0, 2.0)
    starts[1, :, 2] = 1.0
    raw = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32)
    ray = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    points = starts + raw[..., None] * ray

    segments = build_segments(
        points,
        raw,
        np.ones((2, 3), dtype=bool),
        ray,
        surface_margin_m=0.1,
        invalid_depth_ratio=0.5,
        grid_stride=1,
        boundary_padding_cells=0.5,
    )

    endpoints = segments.reshape(-1, 3)
    np.testing.assert_allclose(np.min(endpoints[:, 1]), -3.0, atol=1.0e-6)
    np.testing.assert_allclose(np.max(endpoints[:, 1]), 3.0, atol=1.0e-6)
    np.testing.assert_allclose(np.min(endpoints[:, 2]), -0.5, atol=1.0e-6)
    np.testing.assert_allclose(np.max(endpoints[:, 2]), 1.5, atol=1.0e-6)

    lower_left_depth_line = segments[
        np.isclose(segments[:, 0, 0], 0.0)
        & np.isclose(segments[:, 1, 0], 1.1)
        & np.isclose(segments[:, 0, 1], -3.0)
        & np.isclose(segments[:, 0, 2], -0.5)
    ]
    assert lower_left_depth_line.shape == (1, 2, 3)


def test_hydroshear_roi_visualization_reads_cached_geometry_without_selected_points():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    visualizer_text = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(visualizer_text)
    roi_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "hydroshear_roi_wireframe_arrays"
    )
    roi_source = ast.unparse(roi_node)

    assert "_rl_tacmap_surface_points_w" in roi_source
    assert "_rl_tacmap_surface_raw_m" in roi_source
    assert "_rl_tacmap_surface_valid" in roi_source
    assert "_rl_tacmap_ray_directions_w" in roi_source
    assert "object_sample_roi_surface_margin_m" in roi_source
    assert "object_sample_roi_invalid_depth_ratio" in roi_source
    assert "object_sample_roi_boundary_padding_cells" in roi_source
    assert "_object_sample_points_l" not in roi_source
    assert "_batch_prev_sample_ids" not in roi_source
    assert "_batch_prev_sdf" not in roi_source
    assert "_estimate_object_sample_sdf_batched" not in roi_source
    assert "_select_object_sample_roi_batched" not in roi_source
    assert "--show-hydroshear-roi" in visualizer_text
    assert "--hide-hydroshear-roi" in visualizer_text
    assert "/Visuals/RLTactileObs/HydroShearROI/BoundaryWireframeOrange" in visualizer_text
    assert "/Visuals/RLTactileObs/HydroShearROI/SelectedOrange" not in visualizer_text
    assert "/Visuals/RLTactileObs/HydroShearROI/ContactRed" not in visualizer_text

    training_paths = (
        repo_root / "scripts" / "rsl_rl" / "train.py",
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py",
        repo_root / "integrate" / "curved_hydroshear_adapter.py",
    )
    for path in training_paths:
        assert "show_hydroshear_roi" not in path.read_text(encoding="utf-8")


def test_rl_pressure_heatmaps_share_main_observation_window():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_image", "main"}
    }
    build_names = {node.id for node in ast.walk(functions["build_image"]) if isinstance(node, ast.Name)}
    main_names = {node.id for node in ast.walk(functions["main"]) if isinstance(node, ast.Name)}

    assert "pressure_img" in build_names
    assert "pressure_panels" not in main_names


def test_focus_visuotactile_only_mode_hides_surface_debug_and_orders_focus_panes():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_image", "main"}
    }

    assert '"--focus-visuotactile-only"' in source
    assert 'args_cli.focus_visuotactile_only = str(args_cli.target) == "tacmap"' in source
    assert "args_cli.show_all_tacmap_fingers = False" in source
    assert "args_cli.show_surface_debug = False" in source
    assert "observation_panes = [tacmap_img]" in functions["build_image"]
    assert "observation_panes.append(taxim_image)" in functions["build_image"]
    assert functions["build_image"].index("observation_panes.append(taxim_image)") < functions[
        "build_image"
    ].index("observation_panes.append(hydroshear_img)")
    assert "taxim_image=taxim_image" in functions["main"]


def test_focus_pressure_only_mode_selects_one_named_pad_and_hides_visuotactile_panes():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_image", "main"}
    }

    assert '"--focus-pressure-only"' in source
    assert 'args_cli.focus_pressure_only = str(args_cli.target) == "pressure_pad"' in source
    assert "args_cli.show_taxim_rgb = False" in source
    assert "args_cli.show_hydroshear_marker = False" in source
    assert "args_cli.show_surface_debug = False" in source
    assert "focus_pressure_link = pressure_pad_link_name" in functions["build_image"]
    assert "focus_pressure_img = pressure_panes[pressure_links.index(focus_pressure_link)]" in functions["build_image"]
    assert "image = focus_pressure_img" in functions["build_image"]
    assert "RL Observation — Focus Pressure Pad" in functions["main"]


def test_whole_hand_pressure_only_mode_keeps_all_285_taxels_and_hides_other_modalities():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_image", "main"}
    }

    assert '"--whole-hand-pressure-only"' in source
    assert "args_cli.show_taxim_rgb = False" in source
    assert "args_cli.show_hydroshear_marker = False" in source
    assert "if bool(args_cli.whole_hand_pressure_only):\n        image = pressure_img" in source
    assert "pressure_img = tile_rgb(pressure_panes, cols=6" in functions["build_image"]
    assert "RL Observation — Whole-Hand Pressure (285 Taxels)" in functions["main"]


def test_pressure_visualizer_can_target_the_complete_palm_pad():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    pressure_link_function = next(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pressure_pad_link_name"
    )
    pressure_target_function = next(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pressure_pad_target"
    )

    assert 'PRESSURE_PAD_SEGMENTS = ("mcp", "pip", "palm")' in source
    assert "if str(segment) == 'palm'" in pressure_link_function
    assert "return 'right_hand_rubber_link'" in pressure_link_function
    assert "if str(spec.link_name) == 'right_hand_rubber_link'" in pressure_target_function
    assert "center_index = int(np.argmin" in pressure_target_function
    assert "center_l = points_l[center_index]" in pressure_target_function


def test_visualizer_pressure_uses_opt_in_exact_policy_observation_cache():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"

    observations_tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    ours_node = next(
        node for node in observations_tree.body if isinstance(node, ast.FunctionDef) and node.name == "ours_rl_obs"
    )
    ours_source = ast.unparse(ours_node)
    defaults = {
        arg.arg: default
        for arg, default in zip(ours_node.args.args[-len(ours_node.args.defaults) :], ours_node.args.defaults, strict=True)
    }

    assert isinstance(defaults["cache_pressure_output"], ast.Constant)
    assert defaults["cache_pressure_output"].value is False
    assert "if bool(cache_pressure_output)" in ours_source
    assert "pressure.detach()" in ours_source
    assert "torch.cat((pressure, tacmap_policy, hydroshear), dim=1)" in ours_source

    visualizer_tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: ast.unparse(node)
        for node in visualizer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"configure_pressure_debug", "pressure_maps_from_env", "build_image"}
    }
    assert "cache_pressure_output" in functions["configure_pressure_debug"]
    assert "'cache_pressure_output': True" in functions["configure_pressure_debug"]
    assert '_rl_pressure_observation' in functions["pressure_maps_from_env"]
    assert "policy_observation" in functions["pressure_maps_from_env"]
    assert "pressure_from_policy" in functions["build_image"]
    assert "pressure_flat_cpu = torch.cat(pressure_maps" in functions["build_image"]


def test_visualizer_marker_depth_fusion_uses_opt_in_rl_cache():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"

    observations_tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    hydroshear_node = next(
        node for node in observations_tree.body if isinstance(node, ast.FunctionDef) and node.name == "hydroshear_rl_obs"
    )
    defaults = {
        arg.arg: default
        for arg, default in zip(
            hydroshear_node.args.args[-len(hydroshear_node.args.defaults) :],
            hydroshear_node.args.defaults,
            strict=True,
        )
    }
    hydroshear_source = ast.unparse(hydroshear_node)
    ours_node = next(
        node for node in observations_tree.body if isinstance(node, ast.FunctionDef) and node.name == "ours_rl_obs"
    )
    ours_defaults = {
        arg.arg: default
        for arg, default in zip(
            ours_node.args.args[-len(ours_node.args.defaults) :],
            ours_node.args.defaults,
            strict=True,
        )
    }
    ours_source = ast.unparse(ours_node)

    assert isinstance(defaults["cache_marker_depth_output"], ast.Constant)
    assert defaults["cache_marker_depth_output"].value is False
    assert isinstance(ours_defaults["hydroshear_cache_marker_depth_output"], ast.Constant)
    assert ours_defaults["hydroshear_cache_marker_depth_output"].value is False
    assert "cache_marker_depth_output=hydroshear_cache_marker_depth_output" in ours_source
    assert "marker_ray_depth_grid[:, :sensor_count].detach()" in hydroshear_source
    assert "marker_ray_valid_grid[:, :sensor_count] & calibrated_valid_grid" in hydroshear_source

    visualizer_tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: ast.unparse(node)
        for node in visualizer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "configure_hydroshear_debug",
            "tacmap_marker_joint_inputs",
            "batched_tacmap_image",
        }
    }
    assert "hydroshear_cache_marker_depth_output" in functions["configure_hydroshear_debug"]
    assert "_rl_hydroshear_marker_depth_m" in functions["tacmap_marker_joint_inputs"]
    assert "build_joint_tacmap_interpolation_layout" in functions["tacmap_marker_joint_inputs"]
    assert "joint_marker_depth_values=marker_depth" in functions["batched_tacmap_image"]
    assert "sparse_depth" not in functions["batched_tacmap_image"]


def test_visual_local_tacmap_uses_1000_runtime_rays():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "VISUAL_LOCAL_TACMAP_RAY_COUNT"
    }
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"sample_visual_local_reference", "configure_env"}
    }

    assert ast.literal_eval(assignments["VISUAL_LOCAL_TACMAP_RAY_COUNT"]) == 1000
    assert "image_rows=25" in functions["configure_env"]
    assert "image_cols=40" in functions["configure_env"]
    assert "reference_starts.reshape(-1, 3).index_select(0, selected_indices)" in functions[
        "sample_visual_local_reference"
    ]
    assert "grid_sample" not in functions["sample_visual_local_reference"]
    assert "torch.cdist" not in visualizer_path.read_text(encoding="utf-8")


def test_visual_local_surface_reference_prefers_second_hit_with_first_hit_fallback():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "config"
        / "Revo3"
        / "dexsuite_revo3_env_cfg_grasp.py"
    )
    config_tree = ast.parse(config_path.read_text(encoding="utf-8"))
    make_cfg_source = next(
        ast.unparse(node)
        for node in config_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_tacmap_link_surface_cfg"
    )
    assert "ray_hit_index = 1 if sensor_kind == 'object' else 2" in make_cfg_source
    assert "use_first_hit_fallback=sensor_kind == 'surface'" in make_cfg_source

    sensor_path = repo_root / "tacmap" / "tacmap_sensor" / "sharpa_tacmap_link_surface.py"
    sensor_tree = ast.parse(sensor_path.read_text(encoding="utf-8"))
    sensor_class = next(
        node
        for node in sensor_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SharpaTacmapLinkSurface"
    )
    update_source = next(
        ast.unparse(node)
        for node in sensor_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_buffers_impl"
    )
    assert "second_total = first_depth + eps + second_depth" in update_source
    assert "selected_depth = torch.where(second_valid, second_total" in update_source
    assert "torch.where(fallback_valid, first_depth" in update_source
    assert "raw_ray_depth = export_ray_depth.clone()" in update_source


def test_visual_local_tacmap_points_update_every_environment_step():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "initialize_visual_local_tacmap_refinement",
            "update_visual_local_tacmap_refinement",
            "visual_local_tacmap_world_arrays",
            "visual_local_tacmap_ray_world_arrays",
            "main",
        }
    }

    parser_block_start = source.index('"--tacmap-local-roi-stable-frames"')
    parser_block = source[parser_block_start : source.index(")\n", parser_block_start)]
    assert "default=1" in parser_block
    guard_block_start = source.index('"--tacmap-local-roi-guard-mm"')
    guard_block = source[guard_block_start : source.index(")\n", guard_block_start)]
    assert "default=0.0" in guard_block
    initialize_source = functions["initialize_visual_local_tacmap_refinement"]
    assert "reference_boundary_mask[0, :] = True" in initialize_source
    assert "reference_boundary_mask[-1, :] = True" in initialize_source
    assert "reference_boundary_mask[:, 0] = True" in initialize_source
    assert "reference_boundary_mask[:, -1] = True" in initialize_source
    assert "reference_boundary_mask & reference_valid" in initialize_source

    main_source = functions["main"]
    hide_body_pose = main_source.index("current_body_pose_visualizer.set_visibility(False)")
    env_reset = main_source.index("env.reset()")
    assert hide_body_pose < env_reset
    realtime_update = main_source.index("update_visual_local_tacmap_refinement(env)")
    realtime_points = main_source.index("local_tacmap_point_viz.update", realtime_update)
    realtime_rays = main_source.index("visual_local_tacmap_ray_world_arrays(env)", realtime_points)
    low_rate_guard = main_source.index("if visual_update_due:", realtime_rays)
    low_rate_image = main_source.index("image, stats = build_image", low_rate_guard)
    assert low_rate_guard < low_rate_image
    assert realtime_update < realtime_points < realtime_rays < low_rate_image

    update_source = functions["update_visual_local_tacmap_refinement"]
    assert "bounds_changed = any" in update_source
    assert "candidate_exits_guard" not in update_source
    relayout = update_source.index("set_visual_local_tacmap_layout")
    recompute = update_source.index("sensor.update(0.0, force_recompute=True)")
    read_result = update_source.index("distance_along_normal_raw")
    assert relayout < recompute < read_result
    assert "state['sample_object_depth_m'] = object_depth[:write_count]" in update_source
    assert "state['sample_object_valid'] = object_valid[:write_count]" in update_source

    point_source = functions["visual_local_tacmap_world_arrays"]
    assert "sample_object_valid" not in point_source
    assert "colors[depth_flat > 0.0] = penetration_color" in point_source
    assert "object_valid_flat" not in point_source
    assert "contact_strength" not in point_source

    ray_source = functions["visual_local_tacmap_ray_world_arrays"]
    assert "sample_reference_pixel_indices" in ray_source
    assert "reference_boundary_indices" in ray_source
    assert "sample_object_depth_m" not in ray_source
    assert "sample_object_valid" not in ray_source


def test_visual_local_tacmap_point_colors_distinguish_miss_hit_and_penetration():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "visual_local_tacmap_world_arrays"
    )
    namespace = {
        "np": np,
        "torch": torch,
        "Articulation": object,
        "body_index": lambda _robot, _link_name: 0,
        "quat_apply": lambda _quat, points: points,
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"),
        namespace,
    )

    state = {
        "display_reference_points_l": torch.zeros((4, 3), dtype=torch.float32),
        "valid": torch.tensor([True, True, True, False]),
        "depth_m": torch.tensor([0.0, 0.0, 0.0001, 0.0001]),
        "link_name": "focus_link",
    }
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_state_w=torch.tensor(
                [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
            )
        )
    )
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": robot},
        _rl_visual_local_tacmap_refinement=state,
    )

    points, colors = namespace["visual_local_tacmap_world_arrays"](env)

    assert points.shape == (3, 3)
    np.testing.assert_allclose(
        colors,
        np.asarray(
            [
                [1.0, 0.85, 0.0],
                [1.0, 0.85, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_visual_local_tacmap_ray_display_is_limited_to_two_mm_outside_surface():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    source = visualizer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    ray_source = next(
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "visual_local_tacmap_ray_world_arrays"
    )

    assert "VISUAL_LOCAL_TACMAP_RAY_DISPLAY_LENGTH_M = 0.002" in source
    assert "selected_surface_points = selected_starts + direction * selected_reference_depth.unsqueeze(-1)" in ray_source
    assert "selected_surface_points - direction * float(VISUAL_LOCAL_TACMAP_RAY_DISPLAY_LENGTH_M)" in ray_source
    assert "torch.stack((selected_display_starts, selected_surface_points), dim=1)" in ray_source
    assert "selected_length = torch.where" not in ray_source


def test_visual_local_tacmap_directly_round_trips_integer_reference_pixels():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "visual_local_tacmap_integer_grid_shape",
            "visual_local_tacmap_metric_grid_shape",
            "sample_visual_local_reference",
            "rasterize_visual_local_tacmap_depth",
        }
    }
    namespace = {
        "math": math,
        "torch": torch,
        "VISUAL_LOCAL_TACMAP_RAY_COUNT": 10,
    }
    exec(
        compile(
            ast.Module(body=list(functions.values()), type_ignores=[]),
            str(visualizer_path),
            "exec",
        ),
        namespace,
    )

    rows, cols = 8, 6
    row_grid, col_grid = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    reference_starts = torch.stack(
        (row_grid.to(torch.float32), col_grid.to(torch.float32), torch.zeros((rows, cols))),
        dim=-1,
    )
    reference_depth = (torch.arange(rows * cols, dtype=torch.float32) + 1.0).reshape(rows, cols)
    reference_valid = torch.ones((rows, cols), dtype=torch.bool)
    reference_valid[2, 3] = False
    state = {
        "native_rows": rows,
        "native_cols": cols,
        "reference_starts_l": reference_starts,
        "reference_xy_camera_m": torch.stack(
            (col_grid.to(torch.float32), row_grid.to(torch.float32)),
            dim=-1,
        ),
        "reference_depth_m": reference_depth,
        "reference_valid": reference_valid,
    }

    starts, depth, valid, indices, write_count, roi_valid, grid_shape = namespace[
        "sample_visual_local_reference"
    ](state, (0.0, float(cols - 1), 0.0, float(rows - 1)))

    assert starts.shape == (10, 3)
    assert write_count == 10
    assert len(torch.unique(indices)) == 10
    assert 0 < grid_shape[0] * grid_shape[1] <= 10
    assert torch.equal(starts, reference_starts.reshape(-1, 3).index_select(0, indices))
    assert torch.equal(depth, reference_depth.reshape(-1).index_select(0, indices))
    assert torch.equal(valid, reference_valid.reshape(-1).index_select(0, indices))
    assert torch.equal(roi_valid, reference_valid)

    state.update(
        {
            "sample_reference_pixel_indices": indices,
            "sample_write_count": write_count,
        }
    )
    local_depth = torch.arange(10, dtype=torch.float32) * 0.001
    raster_depth, raster_sampled = namespace["rasterize_visual_local_tacmap_depth"](
        state,
        local_depth,
        valid,
    )
    expected_depth = torch.zeros_like(reference_depth).reshape(-1)
    expected_depth.index_copy_(0, indices, torch.where(valid, local_depth, torch.zeros_like(local_depth)))
    expected_sampled = torch.zeros_like(reference_valid).reshape(-1)
    expected_sampled.index_copy_(0, indices, valid)
    assert torch.equal(raster_depth.reshape(-1), expected_depth)
    assert torch.equal(raster_sampled.reshape(-1), expected_sampled)

    small = namespace["sample_visual_local_reference"](state, (1.0, 2.0, 2.0, 3.0))
    small_indices = small[3]
    assert small[4] == 4
    assert torch.equal(small_indices[:4], torch.tensor([13, 14, 19, 20]))
    assert len(torch.unique(small_indices[:4])) == 4

    metric_rows, metric_cols = namespace["visual_local_tacmap_metric_grid_shape"](
        0.020,
        0.010,
        1000,
        max_rows=480,
        max_cols=640,
    )
    assert 0.98 * 1000 <= metric_rows * metric_cols <= 1000
    assert abs((0.010 / metric_rows) / (0.020 / metric_cols) - 1.0) < 0.05


def test_visual_local_tacmap_applies_sharpa_gaussian_blur_only_in_display_overlay():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"gaussian_blur_visual_tacmap", "overlay_visual_local_tacmap_refinement"}
    }
    namespace = {
        "math": math,
        "torch": torch,
        "VISUAL_LOCAL_TACMAP_GAUSSIAN_KERNEL_SIZE": 9,
        "VISUAL_LOCAL_TACMAP_GAUSSIAN_SIGMA": 1.5,
    }
    exec(
        compile(
            ast.Module(body=[functions["gaussian_blur_visual_tacmap"]], type_ignores=[]),
            str(visualizer_path),
            "exec",
        ),
        namespace,
    )
    source = torch.zeros((17, 17), dtype=torch.uint8)
    source[8, 8] = 255
    before = source.clone()

    blurred = namespace["gaussian_blur_visual_tacmap"](source)

    assert torch.equal(source, before)
    assert blurred.dtype == torch.uint8
    assert 0 < int(blurred[8, 8]) < 255
    assert int(blurred[8, 8]) > int(blurred[8, 9]) > int(blurred[8, 11])
    assert torch.equal(blurred, torch.flip(blurred, dims=(0, 1)))
    assert torch.equal(
        namespace["gaussian_blur_visual_tacmap"](torch.full((17, 17), 64, dtype=torch.uint8)),
        torch.full((17, 17), 64, dtype=torch.uint8),
    )
    sparse = torch.zeros((17, 17), dtype=torch.uint8)
    sampled = torch.zeros((17, 17), dtype=torch.bool)
    sparse[::4, ::4] = 64
    sampled[::4, ::4] = True
    assert torch.equal(
        namespace["gaussian_blur_visual_tacmap"](sparse, sampled),
        torch.full((17, 17), 64, dtype=torch.uint8),
    )

    overlay_source = ast.unparse(functions["overlay_visual_local_tacmap_refinement"])
    assert overlay_source.index("to(torch.uint8)") < overlay_source.index(
        "gaussian_blur_visual_tacmap(local_gray, sampled_display)"
    )

    training_paths = (
        repo_root / "scripts" / "rsl_rl" / "train.py",
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py",
    )
    for path in training_paths:
        assert "gaussian_blur_visual_tacmap" not in path.read_text(encoding="utf-8")


def test_visualizer_joint_tacmap_layout_includes_marker_samples_in_triangulation():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_joint_tacmap_interpolation_layout"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    indices, weights, pixel_valid = namespace["build_joint_tacmap_interpolation_layout"](
        np.asarray([[[0.5, 0.5]]], dtype=np.float32),
        np.asarray([[True]]),
        source_rows=2,
        source_cols=2,
        target_rows=20,
        target_cols=20,
    )

    assert indices.shape == (1, 20, 20, 3)
    assert weights.shape == indices.shape
    assert pixel_valid.shape == (1, 20, 20)
    assert np.all(pixel_valid)
    assert np.any(indices == 4)  # Four base samples occupy 0..3; Marker is sample 4.
    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, atol=1.0e-6)


def test_visualizer_pressure_maps_prefer_exact_policy_values_over_raw_sensor_cache():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pressure_maps_from_env"
    )
    namespace = {
        "torch": torch,
        "TIANJI_PRESSURE_SENSOR_NAMES": ("sensor_a", "sensor_b"),
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    sensor_a = SimpleNamespace(
        data=SimpleNamespace(pressure_force_map=torch.tensor([[[[100.0, 200.0]]]], dtype=torch.float32))
    )
    sensor_b = SimpleNamespace(
        data=SimpleNamespace(pressure_force_map=torch.tensor([[[[300.0, 400.0]]]], dtype=torch.float32))
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=SimpleNamespace(sensors={"sensor_a": sensor_a, "sensor_b": sensor_b}),
        _rl_pressure_observation=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
    )

    maps = namespace["pressure_maps_from_env"](env)

    assert torch.equal(torch.cat(maps, dim=1), env._rl_pressure_observation)
    assert env._rl_pressure_visual_source == "policy_observation"


def test_tile_rgb_can_pad_without_stretching_taxels():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "tile_rgb"
    )
    namespace = {
        "np": np,
        "to_numpy_uint8_rgb": lambda pane: np.asarray(pane, dtype=np.uint8),
        "resize_nearest_to_shape": lambda pane, _height, _width: pane,
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"), namespace)
    small = np.full((4, 4, 3), 32, dtype=np.uint8)
    small[1:3, 1:3] = 255
    tall = np.full((8, 4, 3), 32, dtype=np.uint8)

    tiled = namespace["tile_rgb"]([small, tall], cols=2, gap=0, preserve_scale=True)

    taxel_pixels = np.argwhere(tiled[:, :4, 0] == 255)
    assert np.ptp(taxel_pixels[:, 0]) == np.ptp(taxel_pixels[:, 1])


def test_rl_observation_window_does_not_add_black_placeholder_panes():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"tile_rgb", "build_hydroshear_marker_image"}
    }
    namespace = {
        "np": np,
        "to_numpy_uint8_rgb": lambda pane: np.asarray(pane, dtype=np.uint8),
        "resize_nearest_to_shape": lambda pane, _height, _width: pane,
    }
    exec(compile(ast.Module(body=[functions["tile_rgb"]], type_ignores=[]), str(visualizer_path), "exec"), namespace)

    panes = [np.full((4, 4, 3), 64, dtype=np.uint8)] * 2
    tiled = namespace["tile_rgb"](panes, cols=3, gap=0)
    assert not np.any(np.all(tiled == 0, axis=-1))

    namespace.update(
        {
            "args_cli": SimpleNamespace(show_hydroshear_marker=True, tacmap_scale=1),
            "hydroshear_output_from_env": lambda _env: None,
            "tacmap_display_target_size": lambda: (10, 10),
        }
    )
    exec(
        compile(
            ast.Module(body=[functions["build_hydroshear_marker_image"]], type_ignores=[]),
            str(visualizer_path),
            "exec",
        ),
        namespace,
    )
    assert namespace["build_hydroshear_marker_image"](object()) is None


def test_hydroshear_marker_image_combines_all_fingers_in_camera_frame():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_hydroshear_marker_image"
    )
    images = np.zeros((0, 240, 320, 3), dtype=np.uint8)
    marker_flow = np.zeros((5, 2, 2, 2), dtype=np.float32)
    marker_flow[:, 0, 0] = (40.0, 50.0)
    marker_flow[:, 1, 0] = (40.0, 50.0)
    marker_flow[:, 0, 1] = (80.0, 90.0)
    marker_flow[:, 1, 1] = (84.0, 90.0)
    marker_valid = np.ones((5, 2), dtype=bool)
    render_calls = []

    def render_marker_flow(flow, *, color, draw_points, padding_px):
        assert color == (255, 255, 0)
        assert not draw_points
        assert padding_px == 0
        render_calls.append(np.asarray(flow).copy())
        return np.full((240, 320, 3), 33, dtype=np.uint8)

    namespace = {
        "np": np,
        "args_cli": SimpleNamespace(
            show_hydroshear_marker=True,
            show_all_tacmap_fingers=True,
            focus_finger="index",
        ),
        "FINGER_CHOICES": ("middle", "index", "ring", "pinky", "thumb"),
        "hydroshear_output_from_env": lambda _env: SimpleNamespace(
            marker_images=images,
            marker_flow=marker_flow,
        ),
        "image_from_sensor_batch": lambda _batch, _index: None,
        "hydroshear_marker_pane": lambda image: image,
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"),
        namespace,
    )

    env = SimpleNamespace(
        _brainco_rl_hydroshear_marker_layout=(None, None, None, marker_valid),
        _brainco_rl_hydroshear_adapter=SimpleNamespace(
            _render_vector_field_image=render_marker_flow,
            cfg=SimpleNamespace(marker_radius=3),
        ),
    )
    focus = namespace["build_hydroshear_marker_image"](env)

    assert focus.shape == (240, 320, 3)
    assert len(render_calls) == 1
    assert np.array_equal(render_calls[0], marker_flow[1])
    assert focus[50, 40].tolist() == [0, 0, 0]
    assert focus[90, 80].tolist() == [0, 0, 0]
    assert focus[10, 10].tolist() == [33, 33, 33]


def test_hydroshear_marker_pane_preserves_native_camera_rectangle():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "hydroshear_marker_pane"
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[0, 0] = (1, 0, 0)
    image[0, -1] = (2, 0, 0)
    image[-1, 0] = (3, 0, 0)
    image[-1, -1] = (4, 0, 0)
    namespace = {
        "np": np,
        "args_cli": SimpleNamespace(tacmap_scale=1),
        "tacmap_display_target_size": lambda: (320, 240),
        "resize_nearest_to_shape": lambda source, height, width: np.asarray(source)[:height, :width],
        "to_numpy_uint8_rgb": lambda source: np.asarray(source, dtype=np.uint8),
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"),
        namespace,
    )

    pane = namespace["hydroshear_marker_pane"](image)

    assert pane.shape == (240, 320, 3)
    np.testing.assert_array_equal(pane, image)


def test_hydroshear_marker_pane_uses_markerless_reference_background():
    repo_root = Path(__file__).resolve().parents[1]
    visualizer_path = repo_root / "scripts" / "rsl_rl" / "visualize_rl_tactile_obs.py"
    background_path = (
        repo_root
        / "vitai_4Fingers-320*240"
        / "marker_annotations"
        / "reference_median_markerless.png"
    )
    tree = ast.parse(visualizer_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "hydroshear_marker_pane"
    )
    namespace = {
        "np": np,
        "args_cli": SimpleNamespace(tacmap_scale=1),
        "VITAI_MARKER_BACKGROUND": background_path,
        "tacmap_display_target_size": lambda: None,
        "resize_nearest_to_shape": lambda source, height, width: np.asarray(source)[:height, :width],
        "to_numpy_uint8_rgb": lambda source: np.asarray(source, dtype=np.uint8),
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(visualizer_path), "exec"),
        namespace,
    )
    marker_layer = np.zeros((240, 320, 3), dtype=np.uint8)
    marker_layer[120, 160] = (0, 255, 0)

    pane = namespace["hydroshear_marker_pane"](marker_layer)

    assert np.any(pane[0, 0] != 0)
    np.testing.assert_array_equal(pane[120, 160], marker_layer[120, 160])


def _adaptive_tacmap_observation_helpers():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    names = {
        "_camera_plane_ray_grid_from_rectangle",
        "_local_tacmap_contact_roi_batched",
        "_constant_stride_integer_lattice",
        "_sample_local_tacmap_reference_batched",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np, "torch": torch}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(observations_path), "exec"),
        namespace,
    )
    return namespace


def test_adaptive_tacmap_camera_grid_uses_fixed_white_320_by_240_rectangle():
    namespace = _adaptive_tacmap_observation_helpers()
    starts, directions, row_axis = namespace["_camera_plane_ray_grid_from_rectangle"](
        np.asarray(
            [
                [-2.0, -1.5, 0.0],
                [2.0, -1.5, 0.0],
                [2.0, 1.5, 0.0],
                [-2.0, 1.5, 0.0],
            ],
            dtype=np.float32,
        ),
        camera_origin_link=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        camera_rotation_link=np.eye(3, dtype=np.float32),
        rows=6,
        cols=8,
    )

    assert starts.shape == (6, 8, 3)
    assert directions.shape == starts.shape
    np.testing.assert_allclose(starts[..., 2], 3.0, atol=1.0e-7)
    np.testing.assert_allclose(
        directions,
        np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), directions.shape),
        atol=1.0e-7,
    )
    assert row_axis == 1
    assert np.all(np.diff(starts[0, :, 0]) > 0.0)
    assert np.all(np.diff(starts[:, 0, 1]) > 0.0)
    camera_xy = starts[..., :2]
    np.testing.assert_allclose(np.ptp(camera_xy[..., 0]), 3.5, atol=1.0e-7)
    np.testing.assert_allclose(np.ptp(camera_xy[..., 1]), 2.5, atol=1.0e-7)


def test_adaptive_tacmap_loads_five_shared_white_camera_rectangles():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source/BrainCo_DexHand/BrainCo_DexHand/tasks/manager_based/dexsuite/mdp/observations.py"
    )
    marker_layout_path = (
        repo_root
        / "assets/revo21_right_touch/marker_positions/vitai_4fingers/marker_positions.npz"
    )
    tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    function_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_camera_ray_rectangles"
    )
    namespace = {
        "json": json,
        "np": np,
        "Path": Path,
        "RL_FINGER_ORDER": ("middle", "index", "ring", "pinky", "thumb"),
    }
    exec(
        compile(ast.Module(body=[function_node], type_ignores=[]), str(observations_path), "exec"),
        namespace,
    )

    rectangles, shape, rectangle_path = namespace["_load_camera_ray_rectangles"](
        marker_layout_path
    )

    assert shape == (240, 320)
    assert rectangle_path.name == "camera_ray_rectangles_320x240.json"
    assert tuple(rectangles) == ("middle", "index", "ring", "pinky", "thumb")
    for rectangle in rectangles.values():
        assert rectangle.shape == (4, 3)
        np.testing.assert_allclose(rectangle[:, 2], 0.0, atol=1.0e-9)
        extent = np.ptp(rectangle[:, :2], axis=0)
        np.testing.assert_allclose(extent[0] / extent[1], 320.0 / 240.0, atol=1.0e-7)
    np.testing.assert_allclose(
        rectangles["index"][[0, 2]],
        np.asarray(
            [
                [-0.011144545427, -0.006037417934, 0.0],
                [0.011517734433, 0.010959291037, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1.0e-9,
    )


def test_adaptive_tacmap_roi_is_batched_and_expands_equally_in_camera_xy():
    namespace = _adaptive_tacmap_observation_helpers()
    row, col = torch.meshgrid(
        torch.tensor([-0.6, 0.0, 0.6]),
        torch.tensor([-0.8, 0.0, 0.8, 1.2]),
        indexing="ij",
    )
    coarse_xy = torch.stack((col, row), dim=-1)
    coarse_depth = torch.zeros((2, 3, 4), dtype=torch.float32)
    coarse_depth[0, 1, 1:3] = 0.001
    full_bounds = torch.tensor([-1.0, 1.5, -1.0, 1.0], dtype=torch.float32)

    bounds, normalized, active = namespace["_local_tacmap_contact_roi_batched"](
        coarse_depth,
        coarse_xy,
        full_bounds,
        contact_threshold_m=0.00002,
        roi_margin_m=0.1,
    )

    torch.testing.assert_close(bounds[0], torch.tensor([-0.1, 0.9, -0.1, 0.1]))
    torch.testing.assert_close(bounds[1], full_bounds)
    assert torch.equal(active, torch.tensor([True, False]))
    assert torch.all((normalized[0] >= 0.0) & (normalized[0] <= 1.0))
    assert torch.equal(normalized[1], torch.zeros(4))


def test_adaptive_tacmap_samples_exact_dense_integer_pixels_for_every_environment():
    namespace = _adaptive_tacmap_observation_helpers()
    reference_rows = 8
    reference_cols = 6
    row, col = torch.meshgrid(
        torch.arange(reference_rows, dtype=torch.float32),
        torch.arange(reference_cols, dtype=torch.float32),
        indexing="ij",
    )
    starts = torch.stack((col, row, torch.zeros_like(row)), dim=-1)
    directions = torch.zeros_like(starts)
    directions[..., 2] = 1.0
    depth = (row * 100.0 + col).to(torch.float32)
    valid = torch.ones((reference_rows, reference_cols), dtype=torch.bool)
    state = {
        "reference_rows": reference_rows,
        "reference_cols": reference_cols,
        "reference_starts_l": starts.unsqueeze(0),
        "reference_directions_l": directions.unsqueeze(0),
        "reference_depth_m": depth.unsqueeze(0),
        "reference_valid": valid.unsqueeze(0),
        "full_camera_bounds_m": torch.tensor([[0.0, 5.0, 0.0, 7.0]]),
        "storage_row_axis_camera": (1,),
    }
    roi = torch.tensor([[1.0, 4.0, 1.0, 6.0], [0.0, 5.0, 0.0, 7.0]])

    sampled_starts, sampled_directions, sampled_depth, sampled_valid, indices = namespace[
        "_sample_local_tacmap_reference_batched"
    ](
        state,
        0,
        roi,
        local_rows=2,
        local_cols=3,
    )

    assert sampled_starts.shape == (2, 6, 3)
    assert sampled_directions.shape == sampled_starts.shape
    assert sampled_depth.shape == (2, 6)
    assert sampled_valid.shape == (2, 6)
    assert indices.shape == (2, 6)
    assert torch.all((indices >= 0) & (indices < reference_rows * reference_cols))
    flat_starts = starts.reshape(-1, 3)
    flat_depth = depth.reshape(-1)
    assert torch.equal(sampled_starts, flat_starts[indices])
    assert torch.equal(sampled_depth, flat_depth[indices])


def test_adaptive_tacmap_integer_pixel_lattice_has_constant_row_and_column_spacing():
    namespace = _adaptive_tacmap_observation_helpers()
    reference_rows = 64
    reference_cols = 80
    row, col = torch.meshgrid(
        torch.arange(reference_rows, dtype=torch.float32),
        torch.arange(reference_cols, dtype=torch.float32),
        indexing="ij",
    )
    starts = torch.stack((col, row, torch.zeros_like(row)), dim=-1)
    directions = torch.zeros_like(starts)
    directions[..., 2] = 1.0
    state = {
        "reference_rows": reference_rows,
        "reference_cols": reference_cols,
        "reference_starts_l": starts.unsqueeze(0),
        "reference_directions_l": directions.unsqueeze(0),
        "reference_depth_m": torch.ones((1, reference_rows, reference_cols), dtype=torch.float32),
        "reference_valid": torch.ones((1, reference_rows, reference_cols), dtype=torch.bool),
        "full_camera_bounds_m": torch.tensor(
            [[0.0, float(reference_cols - 1), 0.0, float(reference_rows - 1)]],
            dtype=torch.float32,
        ),
        "storage_row_axis_camera": (1,),
    }
    roi = torch.tensor(
        [
            [10.0, 24.0, 10.0, 16.0],
            [20.0, 48.0, 30.0, 44.0],
        ],
        dtype=torch.float32,
    )

    _, _, _, _, indices = namespace["_sample_local_tacmap_reference_batched"](
        state,
        0,
        roi,
        local_rows=5,
        local_cols=10,
    )

    pixel_grid = indices.reshape(2, 5, 10)
    selected_rows = torch.div(pixel_grid, reference_cols, rounding_mode="floor")
    selected_cols = pixel_grid % reference_cols
    row_steps = selected_rows[:, 1:, 0] - selected_rows[:, :-1, 0]
    col_steps = selected_cols[:, :, 1:] - selected_cols[:, :, :-1]
    assert torch.all(row_steps > 0)
    assert torch.all(col_steps > 0)
    assert torch.all(row_steps == row_steps[:, :1])
    assert torch.all(col_steps == col_steps[:, :, :1])
    assert torch.equal(row_steps[:, 0], torch.tensor([2, 4]))
    assert torch.equal(col_steps[:, 0, 0], torch.tensor([2, 4]))


def test_adaptive_tacmap_landscape_reference_returns_canonical_y_by_x_local_patch():
    namespace = _adaptive_tacmap_observation_helpers()
    reference_rows = 80
    reference_cols = 64
    storage_x, storage_y = torch.meshgrid(
        torch.arange(reference_rows, dtype=torch.float32),
        torch.arange(reference_cols, dtype=torch.float32),
        indexing="ij",
    )
    starts = torch.stack((storage_x, storage_y, torch.zeros_like(storage_x)), dim=-1)
    directions = torch.zeros_like(starts)
    directions[..., 2] = 1.0
    state = {
        "reference_rows": reference_rows,
        "reference_cols": reference_cols,
        "reference_starts_l": starts.unsqueeze(0),
        "reference_directions_l": directions.unsqueeze(0),
        "reference_depth_m": torch.ones((1, reference_rows, reference_cols), dtype=torch.float32),
        "reference_valid": torch.ones((1, reference_rows, reference_cols), dtype=torch.bool),
        "full_camera_bounds_m": torch.tensor(
            [[0.0, float(reference_rows - 1), 0.0, float(reference_cols - 1)]],
            dtype=torch.float32,
        ),
        "storage_row_axis_camera": (0,),
    }

    sampled_starts, _, _, _, indices = namespace["_sample_local_tacmap_reference_batched"](
        state,
        0,
        torch.tensor([[10.0, 28.0, 20.0, 28.0]], dtype=torch.float32),
        local_rows=5,
        local_cols=10,
    )

    local_xy = sampled_starts[0, :, :2].reshape(5, 10, 2)
    assert indices.shape == (1, 50)
    torch.testing.assert_close(local_xy[..., 0], local_xy[:1, :, 0].expand(5, -1))
    torch.testing.assert_close(local_xy[..., 1], local_xy[:, :1, 1].expand(-1, 10))
    x_steps = local_xy[:, 1:, 0] - local_xy[:, :-1, 0]
    y_steps = local_xy[1:, :, 1] - local_xy[:-1, :, 1]
    assert torch.all(x_steps == x_steps[:, :1])
    assert torch.all(y_steps == y_steps[:1, :])
    assert torch.all(x_steps > 0.0)
    assert torch.all(y_steps > 0.0)


def test_rl_tacmap_fixed_surface_rays_are_cached_for_coarse_and_marker_paths():
    repo_root = Path(__file__).resolve().parents[1]
    observations_path = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "tasks"
        / "manager_based"
        / "dexsuite"
        / "mdp"
        / "observations.py"
    )
    tree = ast.parse(observations_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_tacmap_surface_debug_requires_live_update",
            "_tacmap_attached_ray_layouts_match",
            "_tacmap_fixed_surface_reference",
            "tacmap_rl_obs",
            "_hydroshear_marker_ray_measurements",
        }
    }
    coarse_source = ast.unparse(functions["tacmap_rl_obs"])
    marker_source = ast.unparse(functions["_hydroshear_marker_ray_measurements"])
    for source in (coarse_source, marker_source):
        assert "_tacmap_fixed_surface_reference" in source
        assert "surface_reference is None" in source
        assert "hasattr(surface_sensor, 'update')" in source
        assert "object_sensor.update(0.0, force_recompute=True)" in source
        assert "surface_reference['surface_dist_m'].expand" in source
    assert "_tacmap_static_surface_reference" in coarse_source

    namespace = {
        "torch": torch,
        "ManagerBasedRLEnv": object,
        "_warn_once": lambda *_args, **_kwargs: None,
        "_tacmap_raw_image": lambda sensor, env, rows, cols: sensor.raw.reshape(env.num_envs, rows, cols),
        "_tacmap_valid_image": lambda sensor, _name, env, rows, cols: sensor.valid.reshape(
            env.num_envs, rows, cols
        ),
        "_tacmap_vec_image": lambda *_args, **_kwargs: None,
    }
    exec(
        compile(
            ast.Module(
                body=[
                    functions["_tacmap_surface_debug_requires_live_update"],
                    functions["_tacmap_attached_ray_layouts_match"],
                    functions["_tacmap_fixed_surface_reference"],
                ],
                type_ignores=[],
            ),
            str(observations_path),
            "exec",
        ),
        namespace,
    )

    class FakeSensor:
        def __init__(self):
            self.cfg = SimpleNamespace(
                prim_path="/finger",
                offset=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
                debug_viz_link_surfaces=False,
                debug_viz_hits=False,
                debug_viz_rays=False,
            )
            self.ray_starts_att = torch.zeros((2, 4, 3), dtype=torch.float32)
            self.ray_directions_att = torch.zeros((2, 4, 3), dtype=torch.float32)
            self.ray_directions_att[..., 2] = 1.0
            self.raw = torch.tensor(
                [[0.01, 0.02, 0.03, 0.04], [0.011, 0.021, 0.031, 0.041]], dtype=torch.float32
            )
            self.valid = torch.ones((2, 4), dtype=torch.bool)
            self.update_count = 0

        def update(self, _dt, *, force_recompute=False):
            assert force_recompute
            self.update_count += 1

    env = SimpleNamespace(num_envs=2, device="cpu")
    surface_sensor = FakeSensor()
    object_sensor = FakeSensor()
    get_reference = namespace["_tacmap_fixed_surface_reference"]
    first = get_reference(
        env,
        surface_name="surface",
        object_name="object",
        surface_sensor=surface_sensor,
        object_sensor=object_sensor,
        rows=2,
        cols=2,
        require_geometry=False,
    )
    surface_sensor.raw.fill_(0.5)
    second = get_reference(
        env,
        surface_name="surface",
        object_name="object",
        surface_sensor=surface_sensor,
        object_sensor=object_sensor,
        rows=2,
        cols=2,
        require_geometry=False,
    )

    assert first is second
    assert surface_sensor.update_count == 1
    torch.testing.assert_close(
        second["surface_dist_m"],
        torch.tensor([[[0.01, 0.02], [0.03, 0.04]]], dtype=torch.float32),
    )
