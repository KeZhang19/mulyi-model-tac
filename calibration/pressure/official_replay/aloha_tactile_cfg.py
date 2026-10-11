# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch
from pathlib import Path

@dataclass
class TrackInfo:
    rp: object          # RigidPrim instance
    p_rel: torch.Tensor # (3,) target position in RB frame
    q_rel: torch.Tensor # (4,) target quat in RB frame
    rb_path: str


@dataclass
class AlohaTactileEnvCfg:
    """Configuration for the ALOHA tactile inference environment."""

    # Robot
    urdf_path: str = "/home/abc/brainco-description/revo2_system/urdf/revo2_right.urdf"
    robot_prim_path: str = "/World/Robot"
    robot_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_init_rot_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fix_base: bool = True
    press_setup_edit_robot_pose: bool = False
    merge_fixed_joints: bool = False
    urdf_drive_stiffness: float = 400.0
    urdf_drive_damping: float = 40.0
    lock_press_finger_joints: bool = False
    force_urdf_conversion: bool = True
    usd_output_dir: str | None = str(Path(__file__).resolve().parent / "output" / "revo2_urdf")

    # Objects
    enable_plug: bool = True
    enable_socket: bool = False
    asset_root: str = Path(__file__).resolve().parent / "assets"
    automate_asset_id: str = "00186"
    plug_fix_base: bool = False
    socket_fix_base: bool = False
    plug_scale: float = 1.0
    socket_scale: float = 1.0
    plug_collider_type: str = "convex_decomposition"
    socket_collider_type: str = "convex_decomposition"
    force_objects_urdf_conversion: bool = True
    plug_default_pose = (0.0, +0.05, +0.003, 0.0, 0.0, +1.0, 0.0)
    socket_default_pose = (0.0, -0.05, +0.003, 0.0, 0.0, +1.0, 0.0)

    # Tactile sensor
    num_rows: int = 30
    num_cols: int = 30
    point_distance: float = 0.001
    row_distance: float | None = None
    col_distance: float | None = None
    normal_axis: int = 0
    normal_offset: float = 0.0036
    normal_sign: float | None = None
    patch_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    patch_offset_quat: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, -0.7071068)
    stiffness: float = 5_000.0
    damping: float = 0.0
    max_force: float = 10.0
    pressure_gain: float = 1.0
    pressure_bias: float = 0.0
    pressure_gamma: float = 1.0
    pressure_threshold: float = 0.0
    taxel_area: float = 1.0
    show_pressure_pad_centers: bool = False
    pressure_pad_center_marker_radius: float = 0.003
    show_pressure_pad_taxel_points: bool = False
    pressure_pad_taxel_point_radius: float = 0.00045
    pressure_pad_taxel_layout_urdf: str | None = None

    # PhysX contact projection is a sparse sanity/debug channel. It is not a
    # dense pressure-map oracle and should not be used as L1 acceptance GT.
    enable_physx_contact_force_map: bool = False
    physx_contact_usage: str = "L1_sparse_sanity_debug_only"
    physx_contact_max_data_count_per_prim: int = 64
    physx_contact_kernel_sigma: float | None = None
    physx_contact_kernel_radius: float | None = None
    mesh_max_dist: float = 0.20
    mesh_signed: bool = False
    mesh_signed_distance_method: str = "winding"
    mesh_shell_thickness: float = 0.001
    mesh_unsigned_shell_as_contact: bool = True
    mesh_unsigned_contact_mode: str = "normal_ray"
    penetration_deadband: float = 0.0
    pressure_response_model: str = "auto"
    gate_warpsdf_with_physx_contact: bool = False
    warpsdf_contact_gate_usage: str = "experimental_debug_not_l1_acceptance"
    warpsdf_contact_gate_min_contacts: int = 1
    left_arm_target_mesh_prim: str = "/World/Socket"
    right_arm_target_mesh_prim: str = "/World/Plug"
    debug_vis: bool = False
    debug_vis_point_radius: float = 0.0015
    debug_vis_force_threshold: float = 1.0e-6
    debug_vis_show_all_taxels: bool = False
    debug_vis_show_axes: bool = False
    debug_vis_axes_scale: float = 0.03
    tactile_link_keywords: tuple[str, ...] = ("touch_link", "elastomer")
    tactile_sensor_labels: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
    enable_touch_compliant_material: bool = True
    compliant_contact_stiffness: float = 280.0
    compliant_contact_damping: float = 20.0
    touch_contact_offset: float = 0.0002
    touch_rest_offset: float = -0.0001
    touch_collision_paths: tuple[str, ...] = (
        "right_index_touch_link/collisions",
        "right_middle_touch_link/collisions",
        "right_ring_touch_link/collisions",
        "right_pinky_touch_link/collisions",
    )

    # Press test
    enable_press_motion: bool = False
    enable_sample_point_view: bool = False
    press_touch_link: str = "right_middle_touch_link"
    press_touch_links: tuple[str, ...] = ()
    press_link_patch_specs: tuple[
        tuple[str, tuple[float, float, float], tuple[float, float, float]],
        ...,
    ] = ()
    press_points_npy: str = str(
        Path(__file__).resolve().parents[3] / "tacmap" / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy"
    )
    press_normals_npy: str = str(
        Path(__file__).resolve().parents[3] / "tacmap" / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy"
    )
    press_motion_frame: str = "touch"
    press_map_correction_scale: float = 1e-3
    press_center_l: tuple[float, float, float] | None = None
    press_normal_l: tuple[float, float, float] | None = None
    press_patch_pos_l: tuple[float, float, float] | None = None
    press_patch_quat_l: tuple[float, float, float, float] | None = None
    press_tacmap_files_enabled: bool = True
    press_start_offset: float = 0.025
    press_end_offset: float = 0.018
    press_steps: int = 240
    press_slide_distance: float = 0.0
    press_slide_steps: int = 0
    press_slide_axis_l: tuple[float, float, float] = (0.0, 1.0, 0.0)
    press_object_usd_path: str = str(Path(__file__).resolve().parents[3] / "tacmap" / "assets" / "presser" / "cylinder_D4.usd")
    press_object_contact_offset: float = 0.0001
    press_object_rest_offset: float = -0.0005
    press_object_collision_enabled: bool = True
    press_object_mass: float = 0.2
    press_object_control: str = "scripted"
    enable_pressure_pad_presser: bool = False
    pressure_pad_presser_link: str = ""
    pressure_pad_presser_prim_path: str = "/World/PressurePadPresser"
    pressure_pad_presser_usd_path: str = ""
    pressure_pad_presser_scale: float = 1.0
    pressure_pad_presser_start_offset: float = 0.012
    pressure_pad_presser_end_offset: float = -0.001
    pressure_pad_presser_steps: int | None = None
    pressure_pad_presser_offset_l: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pressure_pad_presser_start_offset_l: tuple[float, float, float] | None = None
    pressure_pad_presser_end_offset_l: tuple[float, float, float] | None = None
    press_indent_depth: float | None = None
    press_contact_search_distance: float | None = None
    press_contact_threshold: float = 1.0e-7
    press_hold_joint_pose: str = "zero"
    press_initial_joint_degrees: tuple[tuple[str, float], ...] = ()
    press_motion_actor: str = "object"
    press_hand_axis_link: str = ""
    press_hand_axis_l: tuple[float, float, float] = (1.0, 0.0, 0.0)
    press_finger_joints: tuple[str, ...] = ("right_midmcp_roll_joint",)
    press_finger_start_rad: float = 0.0
    press_finger_end_rad: float = 0.2
    press_finger_kinematic_sensor_pose: bool = True
    press_object_rot_in_touch_frame: tuple[float, float, float, float] = (
        0.7071067811801017,
        -3.019609190115596e-06,
        0.7071067811800986,
        -3.0196091876020132e-06,
    )
    press_object_flip_quat_in_touch_frame: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)
    tacmap_link_surface_ray_axis: str = "+x"
    tacmap_link_surface_ray_direction: tuple[float, float, float] | None = None
    tacmap_link_surface_use_mean_normal: bool = True
    tacmap_link_surface_grid_u_axis: str = "+y"
    tacmap_link_surface_grid_v_axis: str = "+z"
    tacmap_link_surface_grid_center: tuple[float, float, float] = (-0.008, 0.0, 0.0012)

    # Camera / rendering
    enable_camera: bool = True
    camera_width: int = 640
    camera_height: int = 480
    camera_prim_path: str = "/World/Camera"
    camera_eye: tuple[float, float, float] = (-0.9, 0.0, 0.4)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.1)
    set_viewport_camera: bool = True
    viewport_camera_eye: tuple[float, float, float] = (-0.28, -0.22, 0.24)
    viewport_camera_target: tuple[float, float, float] = (-0.005, 0.02, 0.09)
    save_renders: bool = False
    render_output_dir: str = ""

    # Simulation
    physics_dt: float = 1.0 / 120.0
    device: str = "cuda:0"
    headless: bool = True
    enable_physx_gpu_memory_tuning: bool = True
    # PhysX warning:
    # "increase PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity"
    # These defaults add headroom for dense tactile contact scenes.
    physx_gpu_found_lost_aggregate_pairs_capacity: int = 4096
    physx_gpu_total_aggregate_pairs_capacity: int = 8192
    physx_gpu_found_lost_pairs_capacity: int = 8192
    physx_gpu_total_pairs_capacity: int = 262144
    physx_gpu_max_rigid_contact_count: int = 8388608
    physx_gpu_max_rigid_patch_count: int = 1310720


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_JOINT_ORDER = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_thumb_distal_joint",
    "right_index_proximal_joint",
    "right_index_distal_joint",
    "right_middle_proximal_joint",
    "right_middle_distal_joint",
    "right_ring_proximal_joint",
    "right_ring_distal_joint",
    "right_pinky_proximal_joint",
    "right_pinky_distal_joint",
]
