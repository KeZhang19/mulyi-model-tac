from __future__ import annotations

import ast
import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _load_aloha_env_module():
    module_dir = Path(__file__).resolve().parents[1] / "scripts" / "force_map" / "official_replay"
    module_path = module_dir / "aloha_tactile_env.py"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    spec = importlib.util.spec_from_file_location("aloha_tactile_env", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def test_select_press_elastomer_links_prefers_exact_link_basename():
    module = _load_aloha_env_module()

    selected = module._select_press_elastomer_links(
        [
            "/World/Robot/right_midpip_roll_link/right_midpip_roll_touch_link_collision",
            "/World/Robot/right_midpip_roll_touch_link",
        ],
        "right_midpip_roll_touch_link",
    )

    assert selected == ["/World/Robot/right_midpip_roll_touch_link"]


def test_select_press_elastomer_links_falls_back_to_path_match():
    module = _load_aloha_env_module()

    selected = module._select_press_elastomer_links(
        ["/World/Robot/nested/right_midpip_roll_touch_link_collision"],
        "right_midpip_roll_touch_link",
    )

    assert selected == ["/World/Robot/nested/right_midpip_roll_touch_link_collision"]


def test_select_press_elastomer_links_keeps_requested_multi_link_order():
    module = _load_aloha_env_module()

    selected = module._select_press_elastomer_links(
        [
            "/World/Robot/right_middip_roll_rubber_link",
            "/World/Robot/right_indexdip_roll_rubber_link",
            "/World/Robot/right_thumbdip_roll_rubber_link",
        ],
        ("right_thumbdip_roll_rubber_link", "right_middip_roll_rubber_link"),
    )

    assert selected == [
        "/World/Robot/right_thumbdip_roll_rubber_link",
        "/World/Robot/right_middip_roll_rubber_link",
    ]


def test_main_urdf_and_controller_joints_have_unique_names():
    repo_root = Path(__file__).resolve().parents[1]
    urdf = (
        repo_root
        / "assets"
        / "revo21_right_touch"
        / "urdf"
        / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )
    root = ET.parse(urdf).getroot()
    joint_names = [joint.attrib["name"] for joint in root.findall("joint")]

    assert len(joint_names) == len(set(joint_names))

    seen_links = set()
    early_joints = []
    for child in root:
        if child.tag == "link":
            seen_links.add(child.attrib["name"])
        elif child.tag == "joint":
            parent = child.find("parent").attrib["link"]
            link = child.find("child").attrib["link"]
            if parent not in seen_links or link not in seen_links:
                early_joints.append(child.attrib["name"])
    assert early_joints == []

    thumbcmp = root.find("./joint[@name='right_thumbcmp_roll_joint']")
    assert thumbcmp is not None
    assert thumbcmp.find("origin").attrib["xyz"] == "0.00199783670091196 0.0195358719276065 0.0520279736546372"
    assert thumbcmp.find("axis").attrib["xyz"] == "0 0 -1"

    config_path = (
        repo_root
        / "assets"
        / "revo21_right_touch" / "config" / "joint_names_revo21_dv2_urdf_right-touch.SLDASM.yaml"
    )
    payload = config_path.read_text(encoding="utf-8").split(":", 1)[1].strip()
    controller_joint_names = ast.literal_eval(payload)

    assert len(controller_joint_names) == len(set(controller_joint_names))


def test_revo21_thumb_cmr_joint_coordinates_match_sdk():
    repo_root = Path(__file__).resolve().parents[1]
    urdfs = (
        repo_root / "assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf",
        repo_root / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right.urdf",
        repo_root / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Right_visual.urdf",
    )
    for urdf in urdfs:
        joint = ET.parse(urdf).getroot().find("./joint[@name='right_thumbcmr_roll_joint']")

        assert joint.find("origin").attrib["rpy"] == "0 3.14159265358979 0"
        assert joint.find("axis").attrib["xyz"] == "0 -1 0"
        assert joint.find("limit").attrib["lower"] == "-0.523598775598"
        assert joint.find("limit").attrib["upper"] == "1.57079632679"


def test_bimanual_urdf_contains_revo21_rubber_links():
    repo_root = Path(__file__).resolve().parents[1]
    hand = ET.parse(
        repo_root / "assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf"
    ).getroot()
    bimanual = ET.parse(
        repo_root / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Bimanual.urdf"
    ).getroot()

    rubber_links = {link.attrib["name"] for link in hand.findall("link") if "_rubber_link" in link.attrib["name"]}
    bimanual_links = {link.attrib["name"] for link in bimanual.findall("link")}

    assert rubber_links <= bimanual_links


def test_right_hand_camera_extrinsics_are_synced():
    repo_root = Path(__file__).resolve().parents[1]
    expected = {
        "right_middip_camera_joint": (
            "right_middip_roll_link",
            "right_middip_camera_link",
            "0.000422185650486755 0.000230553967918183 0.0140291971472168",
            "0.734798080437383 0 1.57436210878442",
        ),
        "right_thumbdip_camera_joint": (
            "right_thumbdip_roll_link",
            "right_thumbdip_camera_link",
            "-0.0016627 0.014713 -0.0002248",
            "-0.782892653589793 -1.5708 0",
        ),
        "right_index_camera": (
            "right_indexdip_roll_link",
            "right_index_camera",
            "0.000407199894029124 0.000789369590070955 0.0140093130576646",
            "0.732642540505513 0 1.58177569739672",
        ),
        "right_ring_camera_joint": (
            "right_ringdip_roll_link",
            "right_ring_camera_link",
            "0.000400462615055252 -0.000327250087309969 0.0140279116594516",
            "0.733117203824051 0 1.56725770326019",
        ),
        "right_pinky_camera_joint": (
            "right_pinkydip_roll_link",
            "right_pinky_camera_link",
            "0.000370359400855362 -0.000887794003758567 0.0140044430154208",
            "0.729626129382148 0 1.55950529765316",
        ),
    }
    urdfs = (
        repo_root / "assets/revo21_right_touch/urdf/revo21_dv2_urdf_right-touch.SLDASM.urdf",
        repo_root / "assets/urdf/Tianji_Revo3/urdf/Tianji_Revo3_Bimanual.urdf",
    )

    for urdf in urdfs:
        root = ET.parse(urdf).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        for name, (parent, child, xyz, rpy) in expected.items():
            joint = root.find(f"./joint[@name='{name}']")
            assert joint is not None
            assert joint.attrib["type"] == "fixed"
            assert child in links
            assert joint.find("parent").attrib["link"] == parent
            assert joint.find("child").attrib["link"] == child
            assert joint.find("origin").attrib == {"xyz": xyz, "rpy": rpy}
            assert joint.find("mimic") is None


def test_ball_probe_pressure_pad_default_orientation_is_head_down():
    repo_root = Path(__file__).resolve().parents[1]
    module_ast = ast.parse((repo_root / "integrate" / "pressure_calibration_setup.py").read_text(encoding="utf-8"))

    quat = None
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "BALL_PROBE_PRESSURE_PAD_QUAT_WXYZ":
                quat = ast.literal_eval(node.value)

    assert quat == (0.7071067811865476, 0.7071067811865475, 0.0, 0.0)


def test_print_press_coordinates_does_not_enable_pressure_pad_center_markers():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert "cfg.show_pressure_pad_centers = bool(args.show_pressure_pad_centers)" in text
    assert "args.show_pressure_pad_centers or args.print_press_coordinates" not in text


def test_pressure_pad_models_are_explicit_for_integrated_and_rl():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    rl_text = (
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
    ).read_text(encoding="utf-8")
    cfg_text = (repo_root / "scripts" / "force_map" / "official_replay" / "aloha_tactile_cfg.py").read_text(
        encoding="utf-8"
    )
    standalone_text = (repo_root / "scripts" / "force_map" / "run_aloha_force_map.py").read_text(encoding="utf-8")
    sensor_cfg_text = (
        repo_root
        / "source"
        / "BrainCo_DexHand"
        / "BrainCo_DexHand"
        / "force_map"
        / "warp_sdf_tactile_cfg.py"
    ).read_text(encoding="utf-8")

    assert '"surface_gap" if pressure_pad_run else "signed_penetration"' in text
    assert 'cfg.mesh_signed = pressure_contact_model == "signed_penetration"' in text
    assert 'cfg.mesh_unsigned_shell_as_contact = pressure_contact_model == "surface_gap"' in text
    assert '"pressure_backend_id": f"warpsdf_{pressure_contact_model}"' in text
    assert "mesh_use_signed_distance=True" in rl_text
    assert 'mesh_signed_distance_method="winding"' in rl_text
    assert "mesh_signed: bool = False" in cfg_text
    assert "mesh_unsigned_shell_as_contact: bool = True" in cfg_text
    assert 'mesh_unsigned_contact_mode: str = "normal_ray"' in cfg_text
    assert "mesh_use_signed_distance=False" in standalone_text
    assert "mesh_unsigned_shell_as_contact=True" in standalone_text
    assert 'mesh_unsigned_contact_mode="normal_ray"' in standalone_text
    assert "mesh_use_signed_distance: bool = False" in sensor_cfg_text
    assert "mesh_unsigned_shell_as_contact: bool = True" in sensor_cfg_text
    assert 'mesh_unsigned_contact_mode: str = "normal_ray"' in sensor_cfg_text


def test_integrated_middle_finger_uses_revo21_tactile_link():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert '"middle": _finger_map(' in text
    assert '"right_middip_roll_rubber"' in text
    assert '"touch_link": f"{stem}_link"' in text
    assert '"points": _REVO21_TACMAP_DIR / f"{stem}_point.npy"' in text
    assert '"normals": _REVO21_TACMAP_DIR / f"{stem}_normal.npy"' in text


def test_integrated_revo21_finger_maps_cover_other_dip_rubber_links():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert 'FINGER_CHOICES = ("middle", "index", "ring", "pinky", "thumb")' in text
    assert "def _finger_map(" in text
    for finger in ("index", "ring", "pinky", "thumb"):
        assert f'"{finger}": _finger_map(' in text
        assert f'"right_{finger}dip_roll_rubber"' in text

    assert "right_thumb_touch_link" not in text
    assert "right_thumb_touch_point.npy" not in text


def test_integrated_supports_comma_separated_fingers_arg():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument(\n    "--fingers"' in text
    assert 'parser.add_argument(\n    "--focus-finger"' in text
    assert "def parse_fingers_arg(" in text
    assert "def resolve_focus_finger(" in text
    assert "def apply_fingers_cfg(" in text
    assert "cfg.press_touch_links = tuple(touch_links)" in text
    assert "primary_finger = resolve_focus_finger(active_fingers, args.focus_finger)" in text
    assert "apply_fingers_cfg(cfg, active_fingers, primary_finger=primary_finger)" in text
    assert 'raise ValueError(f"--focus-finger {focus!r} must also be listed in --fingers {active_fingers!r}")' in text


def test_integrated_tactile_sensor_override_keeps_base_signature():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "integrated_tactile_env.py").read_text(encoding="utf-8")
    block = text[text.index("def _create_tactile_sensors(") : text.index("if str(getattr(cfg, \"tacmap_ray_mode\"")]

    assert "target_root_paths: list[str]," in block
    assert "target_query_paths: list[str]," in block
    assert "target_root_paths,\n            target_query_paths," in block


def test_integrated_demo_defaults_stay_on_legacy_press_path():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert 'default=None,\n    help=(\n        "Source for the scripted presser center/normal.' in text
    assert '"pressure_layout" if pressure_pad_run else "legacy_finger_map"' in text
    assert "args.tacmap_max_distance = 0.015" in text
    assert 'grid_u_size=0.0202' in text
    assert 'grid_v_size=0.0182' in text
    assert 'grid_center=(0.00159, 0.00103, 0.04757)' in text
    assert "args.link_surface_grid_u_size = 0.0202" not in text
    assert "args.link_surface_grid_v_size = 0.0182" not in text
    assert "args.link_surface_grid_center = (0.00159, 0.00103, 0.04757)" not in text
    assert 'args.tacmap_ray_mode = "link_surface"' in text
    assert 'parser.add_argument("--fots-view", choices=("flow", "markers", "both"), default="markers")' in text


def test_tacmap_strip_keeps_red_contact_center_ring():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    block = text[text.index("def tacmap_strip(") : text.index("def image_strip(")]

    assert "pressure_center = (" in block
    assert "_depth_weighted_center_px(tacmap_raw[i])" in block
    assert "_draw_ring(" in block


def test_hydroshear_debug_visuals_keeps_integrated_panels_visible():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    debug_only_block = text[text.index('parser.add_argument(\n    "--hydroshear-debug-only"') : text.index('parser.add_argument("--hydroshear-debug-visuals"')]

    assert "--hydroshear-debug-visuals" not in debug_only_block
    assert 'parser.add_argument("--hydroshear-debug-visuals", "--hydroshear_debug_visuals", action="store_true")' in text
    assert "if args.hydroshear_debug_only or args.hydroshear_debug_visuals:" in text
    assert "hydroshear_output.original_marker_images" in text
    assert "hydroshear_original_pane,\n                hydroshear_pane," in text
    assert 'parser.add_argument("--show-hydroshear-marker-axes", "--show_hydroshear_marker_axes", action="store_true")' in text
    assert "args.show_hydroshear_marker_axes = True" in text
    assert 'path="/Visuals/HydroShearMarkers/env_0/MarkerColAxesCyan"' in text
    assert 'path="/Visuals/HydroShearMarkers/env_0/MarkerRowAxesOrange"' in text


def test_tacmap_finger_marker_viz_uses_calibrated_vitai_layout_and_each_link_pose():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    block = text[text.index("class TacMapFingerMarkerPointViz") : text.index("class HydroShearVectorViz")]
    normal_block = text[text.index("class TacMapFingerMarkerNormalViz") : text.index("class HydroShearVectorViz")]

    assert '"--show-tacmap-finger-marker-points",' in text
    assert "args.show_tacmap_finger_marker_points = True" in text
    assert "_REVO21_VITAI_MARKER_LAYOUT" in text
    assert "class TacMapFingerLinkVizBase" in text
    assert "class TacMapFingerMarkerPointViz(TacMapFingerLinkVizBase)" in text
    assert "class TacMapFingerMarkerNormalViz(TacMapFingerLinkVizBase)" in text
    assert 'points_key = f"{finger}_points_link_m"' in block
    assert 'method_key = f"{finger}_method"' in block
    assert 'exact = finite & distortion_valid & (methods == "ray_hit")' in block
    assert "points = marker_points_l[exact]" in block
    assert '"quality": "high_confidence"' in block
    assert "MagentaApprox" not in block
    assert "np.linalg.norm(marker_points_l, axis=-1) > 1.0e-9" in block
    assert "marker_normals_l / np.maximum(normal_norm[:, None], 1.0e-9) * self.radius" not in block
    assert "body_link_state_w" in text
    assert "pose[:3][None, :] + quat_apply_np(pose[3:7], points_l)" in block
    assert 'root_path="/Visuals/TacMapFingerMarkers/env_0"' in text
    assert '"MarkerPointsBlue"' in block
    assert "sensor_index=display_sensor_index" in text
    assert "finger in FINGER_MAPS" in text
    assert "fingers=tacmap_marker_overlay_fingers" in text
    assert "tacmap_finger_marker_point_viz.update(env)" in text
    assert "class TacMapFingerMarkerNormalViz" in text
    assert 'normals_key = f"{finger}_normals_link"' in normal_block
    assert 'method_key = f"{finger}_method"' in normal_block
    assert 'exact = valid & distortion_valid & (methods == "ray_hit")' in normal_block
    assert "marker_points_l = marker_points_l[exact]" in normal_block
    assert "marker_normals_l = np.asarray(layout[normals_key], dtype=np.float32)" in normal_block
    assert "normals_w = quat_apply_np(pose[3:7], normals_l[:count])" in normal_block
    assert '"MarkerNormalsYellow"' in normal_block
    assert "tacmap_finger_marker_normal_viz.update(env)" in text
    assert "tacmap-finger-point-max" not in text
    assert "tacmap_surface_points_w" not in block
    assert "debug_marker_points_w" not in block


def test_tacmap_finger_center_arrows_show_all_active_finger_link_surface_centers():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    block = text[text.index("class TacMapFingerCenterArrowViz") : text.index("class HydroShearVectorViz")]

    assert 'parser.add_argument("--hide-tacmap-finger-center-arrows", action="store_true")' in text
    assert "class TacMapFingerCenterArrowViz" in text
    assert "class TacMapFingerCenterArrowViz(TacMapFingerLinkVizBase)" in text
    assert '"center_l": center_l.astype(np.float32)' in block
    assert '"direction_l": direction_l.astype(np.float32)' in block
    assert "center_w = pose[:3] + quat_apply_np(pose[3:7], center_l)[0]" in block
    assert "direction_w = _normalize_vec(quat_apply_np(pose[3:7], direction_l)[0])" in block
    assert "path_start_w = center_w + direction_w * self.press_start_offset" in block
    assert "path_end_w = center_w + direction_w * self.press_end_offset" in block
    assert 'path="/Visuals/TacMapFingerCenters/env_0/CenterRayGreen"' in text
    assert "press_start_offset=float(cfg.press_start_offset)" in text
    assert "press_end_offset=float(cfg.press_end_offset)" in text
    assert "fingers=active_fingers" in text
    assert "tacmap_finger_center_arrow_viz.update(env)" in text


def test_integrated_panes_are_single_display_sensor_for_multi_finger_runs():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert "def sensor_display_batch(" in text
    assert "display_finger = primary_finger" in text
    assert "display_sensor_index = active_fingers.index(display_finger)" in text
    assert "force_strip(display_force_pane" in text
    assert "tacmap_display = sensor_display_batch(tacmap, display_sensor_index)" in text
    assert "sensor_display_batch(fots_output.marker_overlay_images, display_sensor_index)" in text
    assert "sensor_display_batch(tacex_rgb_output.tactile_rgb, display_sensor_index)" in text
    assert "sensor_display_batch(tacsl_shear_output.shear_images, display_sensor_index)" in text
    assert "hydroshear_debug_image(hydroshear_output, sensor_index=display_sensor_index)" in text


def test_tacmap_contact_center_visual_is_kept():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument("--hide-tacmap-center-arrow", action="store_true")' in text
    assert "class TacMapCenterArrowViz" in text
    assert 'path="/Visuals/TacMapContactCenter/env_0/CenterNormalRed"' in text
    assert "tacmap_center_arrow_viz.update(" in text
    assert "normal_force_arrow=on " in text


def test_fingertip_warpsdf_pane_is_blank_placeholder():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert "display_force_pane = sensor_display_batch(display_force, display_sensor_index)" in text
    assert "f_img = force_strip(display_force_pane" in text
    assert "if not pressure_pad_run:\n            f_img = force_strip(np.zeros_like(display_force_pane)" in text
    assert "stats_parts = []\n        if pressure_pad_run:" in text


def test_touch_compliance_defaults_to_on_unless_explicitly_disabled():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert "cfg.enable_touch_compliant_material = not bool(args.disable_touch_compliant_material)" in text


def test_pressure_layout_uses_selected_touch_link_for_tactile_discovery():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")
    block = text[text.index("def apply_pressure_layout_urdf_cfg(") : text.index("def apply_presser_cfg(")]

    assert "cfg.press_touch_link = spec.link_name" in block
    assert "cfg.tactile_link_keywords = (spec.link_name,)" in block
    assert 'cfg.touch_collision_paths = (f"{spec.link_name}/collisions",)' in block
    assert "_is_fingertip_pressure_link(spec.link_name)" in block
    assert "fingertips are visual tactile channels, not pressure pads" in block


def test_pressure_pad_center_links_exclude_palm_links():
    module = _load_aloha_env_module()
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = (
        repo_root
        / "assets"
        / "revo21_right_touch" / "urdf" / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )

    links = module._pressure_pad_center_links_from_urdf(urdf_path)

    assert len(links) == 11
    assert "right_hand_rubber_link" in links
    assert all("palm" not in link.lower() for link in links)
    assert all("tip" not in link.lower() for link in links)
    assert all(link.endswith("_touch_link") for link in links if link != "right_hand_rubber_link")


def test_pressure_pad_surface_center_uses_urdf_normal_axis():
    module = _load_aloha_env_module()
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = (
        repo_root
        / "assets"
        / "revo21_right_touch" / "urdf" / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )

    axes = module._pressure_pad_surface_axis_by_link_from_urdf(urdf_path)
    origins = module._pressure_pad_origin_by_link_from_urdf(urdf_path)

    assert axes["right_midpip_roll_touch_link"] == (0, 1.0)
    assert axes["right_indexpip_roll_touch_link"] == (0, 1.0)
    assert axes["right_thumbmcp_roll_touch_link"] == (0, -1.0)
    assert set(origins) == set(module._pressure_pad_center_links_from_urdf(urdf_path))
    assert origins["right_indexpip_roll_touch_link"] == (0.01016522, 0.00056174, 0.01676985)
    assert origins["right_thumbmcp_roll_touch_link"] == (-0.01275706, 0.0295127, 0.0)
    assert origins["right_thumbpip_roll_touch_link"] == (-0.00973271, 0.01680773, 0.00000045)
    assert module._bbox_surface_center((0, 0, 0), (2, 4, 6), normal_axis=0, normal_sign=1) == (2.0, 2.0, 3.0)
    assert module._bbox_surface_center((0, 0, 0), (2, 4, 6), normal_axis=2, normal_sign=-1) == (1.0, 2.0, 0.0)


def test_pressure_pad_taxel_points_cover_declared_urdf_layouts():
    module = _load_aloha_env_module()
    repo_root = Path(__file__).resolve().parents[1]
    urdf_path = (
        repo_root
        / "assets"
        / "revo21_right_touch" / "urdf" / "revo21_dv2_urdf_right-touch.SLDASM.urdf"
    )

    points_by_link = module._pressure_pad_taxel_points_by_link_from_urdf(urdf_path)
    midpip = points_by_link["right_midpip_roll_touch_link"]

    assert set(points_by_link) == set(module._pressure_pad_center_links_from_urdf(urdf_path))
    assert len(points_by_link) == 11
    assert all(len(points) == 32 for points in points_by_link.values())
    assert min(point[1] for point in midpip) < 0.00010799 < max(point[1] for point in midpip)
    assert min(point[2] for point in midpip) < 0.01694991 < max(point[2] for point in midpip)


def test_pressure_pad_taxel_points_default_on_for_legacy_runs():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "integrate" / "run_integrated_tactile.py").read_text(encoding="utf-8")

    assert 'parser.add_argument(\n    "--show-pressure-pad-taxel-points"' in text
    assert 'parser.add_argument(\n    "--hide-pressure-pad-taxel-points"' in text
    assert "cfg.show_pressure_pad_taxel_points = bool(" in text
    assert "(legacy_integrated_run or args.show_pressure_pad_taxel_points) and not args.hide_pressure_pad_taxel_points" in text
    assert "DEFAULT_PRESSURE_PAD_LAYOUT_URDF" in text


def test_pressure_sensor_labels_use_selected_touch_link_not_legacy_finger():
    module = _load_aloha_env_module()
    env = type(
        "Env",
        (),
        {
            "_selected_links": ["/World/Robot/right_thumbpip_roll_touch_link"],
            "_sensor_slot_order": [0],
        },
    )()

    assert module.pressure_sensor_labels_from_env(env, 1, fallback_label="middle") == [
        "right_thumbpip_roll_touch_link"
    ]


def test_press_depth_summary_uses_contact_pose_delta_for_thumb_axis():
    module = _load_aloha_env_module()
    env = type(
        "Env",
        (),
        {
            "_cfg": type("Cfg", (), {"press_indent_depth": 0.003})(),
            "_press_contact_detected": True,
            "_press_contact_pose_w": module.torch.tensor([-0.014858, 0.080859, 0.030000, 1.0, 0.0, 0.0, 0.0]),
            "_press_object_pose_w": module.torch.tensor([-0.011858, 0.080859, 0.030000, 1.0, 0.0, 0.0, 0.0]),
            "_press_object_manual_base_pose_w": module.torch.tensor(
                [-0.038331, 0.080859, 0.030000, 1.0, 0.0, 0.0, 0.0]
            ),
            "_press_debug_axis_w": module.torch.tensor([1.0, 0.0, 0.0]),
        },
    )()

    summary = module.press_depth_summary_from_env(
        env,
        module.np.array([[[1.46851398e-05]]], dtype=module.np.float32),
    )

    assert summary["phase"] == "indent"
    assert module.np.isclose(summary["cmd_indent_m"], 0.003)
    assert module.np.isclose(summary["indent_depth_m"], 0.003)
    assert module.np.isclose(summary["lateral_error_m"], 0.0)
    assert module.np.isclose(summary["sdf_penetration_max_m"], 1.46851398e-05)
    assert module.np.allclose(summary["indent_delta_w_m"], [0.003, 0.0, 0.0])


def test_ball_probe_default_press_axis_follows_probe_local_minus_y():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)

    class DummyMathUtils:
        @staticmethod
        def quat_apply(quat, vec):
            q_xyz = quat[:, 1:4]
            q_w = quat[:, 0].unsqueeze(1)
            uv = module.torch.cross(q_xyz, vec, dim=1)
            uuv = module.torch.cross(q_xyz, uv, dim=1)
            return vec + 2.0 * (q_w * uv + uuv)

    env._cfg = type(
        "Cfg",
        (),
        {
            "press_hand_axis_link": "",
            "press_object_usd_path": "/tmp/ball_probe.usd",
        },
    )()
    env._device = module.torch.device("cpu")
    env._math_utils = DummyMathUtils()
    env._press_object_pose_w = None
    env._press_touch_normal_l = module.torch.tensor((1.0, 0.0, 0.0))

    env._press_object_manual_base_pose_w = module.torch.tensor(
        (0.0, 0.0, 0.0, 0.7071068, 0.0, 0.0, 0.7071068)
    )
    axis = module.AlohaTactileEnv._press_object_axis_w(env, module.torch.tensor((1.0, 0.0, 0.0, 0.0)))

    assert module.np.allclose(axis.detach().cpu().numpy(), [1.0, 0.0, 0.0], atol=1.0e-6)
    assert env._press_debug_axis_source == "ball_probe_local_-y"


def test_capture_press_object_pose_does_not_rewrite_usd_stage():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)
    pose_w = module.np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=module.np.float32)
    sync_calls = []

    class DummyPlug:
        def __init__(self):
            self.data = type("Data", (), {"default_root_state": module.torch.zeros((1, 13))})()
            self.pose = None
            self.velocity = None
            self.update_dt = None

        def write_root_pose_to_sim(self, pose):
            self.pose = pose.detach().clone()

        def write_root_velocity_to_sim(self, velocity):
            self.velocity = velocity.detach().clone()

        def update(self, dt):
            self.update_dt = dt

    plug = DummyPlug()
    env._plug_obj = plug
    env._device = module.torch.device("cpu")
    env._cfg = type("Cfg", (), {"physics_dt": 1.0 / 60.0})()
    env._press_prev_object_pos_w = None
    env._read_press_object_usd_pose = lambda: pose_w.copy()
    env._sync_press_object_usd_pose = lambda *args: sync_calls.append(args)

    captured = module.AlohaTactileEnv.capture_press_object_pose_as_motion_start(env)

    assert module.np.allclose(captured, pose_w)
    assert plug.pose is not None
    assert plug.velocity is not None
    assert plug.update_dt == 0.0
    assert sync_calls == []


def test_manual_gui_press_object_sync_writes_stage_pose_to_sim():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)
    pose_w = module.np.array([0.4, 0.5, 0.6, 0.7071068, 0.0, 0.7071068, 0.0], dtype=module.np.float32)

    class DummyPlug:
        def __init__(self):
            self.data = type("Data", (), {"default_root_state": module.torch.zeros((1, 13))})()
            self.pose = None
            self.velocity = None
            self.update_dt = None

        def write_root_pose_to_sim(self, pose):
            self.pose = pose.detach().clone()

        def write_root_velocity_to_sim(self, velocity):
            self.velocity = velocity.detach().clone()

        def update(self, dt):
            self.update_dt = dt

    plug = DummyPlug()
    env._plug_obj = plug
    env._device = module.torch.device("cpu")
    env._cfg = type(
        "Cfg",
        (),
        {"enable_press_motion": True, "press_object_control": "manual_gui", "physics_dt": 1.0 / 60.0},
    )()
    env._press_prev_object_pos_w = None
    env._read_press_object_usd_pose = lambda: pose_w.copy()

    module.AlohaTactileEnv._sync_manual_gui_press_object_pose_from_stage(env)

    assert plug.pose is not None
    assert module.np.allclose(plug.pose.detach().cpu().numpy()[0, :7], pose_w, atol=1.0e-6)
    assert plug.velocity is not None
    assert plug.update_dt == 0.0


def test_thumb_mcp_ball_probe_setup_uses_measured_initial_y():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)
    pose_w = module.torch.tensor((-0.037722, 0.050977, 0.030000, 1.0, 0.0, 0.0, 0.0))
    written = {}

    class DummyPlug:
        def __init__(self):
            self.update_dt = None

        def update(self, dt):
            self.update_dt = dt

    env._plug_obj = DummyPlug()
    env._cfg = type(
        "Cfg",
        (),
        {
            "press_touch_link": "right_thumbmcp_roll_touch_link",
            "press_object_usd_path": "/tmp/ball_probe.usd",
        },
    )()
    env._press_object_pose_w = pose_w
    env._write_press_object_state = lambda pos, quat: written.setdefault(
        "pose", module.torch.cat([pos.detach().clone(), quat.detach().clone()])
    )
    env._sync_press_object_usd_pose = lambda pos, quat: written.setdefault(
        "usd", module.torch.cat([pos.detach().clone(), quat.detach().clone()])
    )

    prepared = module.AlohaTactileEnv.prepare_press_object_manual_setup(env)

    assert module.np.isclose(float(prepared[1]), 0.042644)
    assert module.np.isclose(float(written["pose"][1]), 0.042644)
    assert module.np.isclose(float(written["usd"][1]), 0.042644)
    assert env._plug_obj.update_dt == 0.0


def test_ball_probe_object_press_moves_down_in_world_z():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)

    class DummyPlug:
        def __init__(self):
            self.data = type("Data", (), {"default_root_state": module.torch.zeros((1, 13))})()
            self.pose = None
            self.velocity = None

        def write_root_pose_to_sim(self, pose):
            self.pose = pose.detach().clone()

        def write_root_velocity_to_sim(self, velocity):
            self.velocity = velocity.detach().clone()

    class DummyMathUtils:
        @staticmethod
        def quat_apply(quat, vec):
            q_xyz = quat[:, 1:4]
            q_w = quat[:, 0].unsqueeze(1)
            uv = module.torch.cross(q_xyz, vec, dim=1)
            uuv = module.torch.cross(q_xyz, uv, dim=1)
            return vec + 2.0 * (q_w * uv + uuv)

        @staticmethod
        def quat_mul(a, b):
            return a

        @staticmethod
        def quat_inv(q):
            out = q.clone()
            out[:, 1:4] = -out[:, 1:4]
            return out

    env._plug_obj = DummyPlug()
    env._robot = type(
        "Robot",
        (),
        {
            "data": type(
                "Data",
                (),
                {
                    "body_link_state_w": module.torch.tensor(
                        [[[0.0, 0.0, 0.0, 0.7071068, 0.0, -0.7071068, 0.0]]]
                    )
                },
            )()
        },
    )()
    env._math_utils = DummyMathUtils()
    env._device = module.torch.device("cpu")
    env._cfg = type(
        "Cfg",
        (),
        {
            "press_steps": 2,
            "press_start_offset": 0.010,
            "press_end_offset": 0.007,
            "press_slide_steps": 0,
            "press_slide_distance": 0.0,
            "press_hand_axis_link": "touch_nearest_world_z",
            "press_object_control": "scripted",
            "physics_dt": 1.0 / 60.0,
        },
    )()
    env._press_counter = 1
    env._press_touch_body_idx = 0
    env._press_touch_center_l = module.torch.zeros(3)
    env._press_touch_normal_l = module.torch.tensor((0.0, 1.0, 0.0))
    env._press_object_manual_base_pose_w = module.torch.tensor((1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
    env._press_hand_axis_body_idx = None
    env._press_hand_axis_l = module.torch.tensor((0.0, 0.0, -1.0))
    env._press_slide_axis_l = None
    env._manual_press_offset_m = None
    env._manual_press_slide_offset_l = None
    env._press_prev_object_pos_w = None
    env._press_object_rot_l = module.torch.tensor((1.0, 0.0, 0.0, 0.0))
    env._press_object_flip_l = module.torch.tensor((1.0, 0.0, 0.0, 0.0))

    module.AlohaTactileEnv._write_press_object_pose(env)

    assert module.np.allclose(env._plug_obj.pose.detach().cpu().numpy()[0, :3], [1.0, 2.0, 2.997])


def test_press_indent_depth_starts_after_warpsdf_contact_onset():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)

    class DummyPlug:
        def __init__(self):
            self.data = type("Data", (), {"default_root_state": module.torch.zeros((1, 13))})()
            self.pose = None
            self.velocity = None

        def write_root_pose_to_sim(self, pose):
            self.pose = pose.detach().clone()

        def write_root_velocity_to_sim(self, velocity):
            self.velocity = velocity.detach().clone()

    class DummyMathUtils:
        @staticmethod
        def quat_apply(quat, vec):
            q_xyz = quat[:, 1:4]
            q_w = quat[:, 0].unsqueeze(1)
            uv = module.torch.cross(q_xyz, vec, dim=1)
            uuv = module.torch.cross(q_xyz, uv, dim=1)
            return vec + 2.0 * (q_w * uv + uuv)

        @staticmethod
        def quat_mul(a, b):
            return a

        @staticmethod
        def quat_inv(q):
            out = q.clone()
            out[:, 1:4] = -out[:, 1:4]
            return out

    env._plug_obj = DummyPlug()
    env._robot = type(
        "Robot",
        (),
        {
            "data": type(
                "Data",
                (),
                {
                    "body_link_state_w": module.torch.tensor(
                        [[[0.0, 0.0, 0.0, 0.7071068, 0.0, -0.7071068, 0.0]]]
                    )
                },
            )()
        },
    )()
    env._math_utils = DummyMathUtils()
    env._device = module.torch.device("cpu")
    env._cfg = type(
        "Cfg",
        (),
        {
            "enable_press_motion": True,
            "press_motion_actor": "object",
            "press_steps": 4,
            "press_start_offset": 0.010,
            "press_end_offset": 0.000,
            "press_slide_steps": 0,
            "press_slide_distance": 0.0,
            "press_hand_axis_link": "touch_nearest_world_z",
            "press_object_control": "scripted",
            "press_indent_depth": 0.002,
            "press_contact_search_distance": 0.009,
            "press_contact_threshold": 1.0e-7,
            "physics_dt": 1.0 / 60.0,
        },
    )()
    env._step_count = 1
    env._press_counter = 1
    env._press_touch_body_idx = 0
    env._press_touch_center_l = module.torch.zeros(3)
    env._press_touch_normal_l = module.torch.tensor((0.0, 1.0, 0.0))
    env._press_object_manual_base_pose_w = module.torch.tensor((1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
    env._press_hand_axis_body_idx = None
    env._press_hand_axis_l = module.torch.tensor((0.0, 0.0, -1.0))
    env._press_slide_axis_l = None
    env._manual_press_offset_m = None
    env._manual_press_slide_offset_l = None
    env._press_prev_object_pos_w = None
    env._press_object_rot_l = module.torch.tensor((1.0, 0.0, 0.0, 0.0))
    env._press_object_flip_l = module.torch.tensor((1.0, 0.0, 0.0, 0.0))
    env._press_contact_pose_w = None
    env._press_contact_counter = None
    env._press_contact_detected = False

    module.AlohaTactileEnv._write_press_object_pose(env)
    contact_candidate_z = float(env._plug_obj.pose.detach().cpu().numpy()[0, 2])
    module.AlohaTactileEnv._update_press_contact_zero(
        env,
        {"pressure_penetration_map": module.np.array([[[1.0e-6]]], dtype=module.np.float32)},
    )
    env._press_counter = 3
    module.AlohaTactileEnv._write_press_object_pose(env)

    assert module.np.isclose(float(env._press_contact_pose_w.detach().cpu().numpy()[2]), contact_candidate_z)
    assert module.np.isclose(float(env._plug_obj.pose.detach().cpu().numpy()[0, 2]), contact_candidate_z - 0.002)


def test_press_object_actor_does_not_create_joint_hold_pose():
    module = _load_aloha_env_module()
    env = object.__new__(module.AlohaTactileEnv)

    class DummyRobot:
        joint_names = ["right_thumbcmp_roll_joint", "right_thumbcmr_roll_joint", "right_midmcp_roll_joint"]

        def __init__(self):
            self.data = type(
                "Data",
                (),
                {
                    "default_joint_pos": module.torch.zeros((1, 3)),
                    "default_joint_vel": module.torch.zeros((1, 3)),
                },
            )()

    cfg = type(
        "Cfg",
        (),
        {
            "enable_press_motion": True,
            "press_motion_actor": "object",
            "press_hold_joint_pose": "zero",
            "press_initial_joint_degrees": (("right_thumbcmr_roll_joint", 80.0),),
        },
    )()
    env._cfg = cfg
    env._robot = DummyRobot()
    env._press_hold_joint_ids = []
    env._press_hold_joint_pos = None
    env._press_hold_joint_vel = None

    module.AlohaTactileEnv._setup_press_finger_motion(env, cfg)

    assert env._press_hold_joint_ids == []
    assert env._press_hold_joint_pos is None
    assert env._press_hold_joint_vel is None


def test_press_only_dataset_mapping_fallback_hold_requires_lock_flag():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "scripts" / "force_map" / "official_replay" / "aloha_tactile_env.py").read_text(
        encoding="utf-8"
    )
    block = text[text.index("# Resolve joint mapping (dataset order). Press-only runs") : text.index("self._setup_press_finger_motion(cfg)")]

    assert 'if bool(getattr(cfg, "lock_press_finger_joints", False)):' in block
    assert 'self._press_hold_joint_ids = list(range(len(self._robot.joint_names)))' in block
    assert "robot joints are not held. Pass --lock-press-finger-joints to hold them." in block
