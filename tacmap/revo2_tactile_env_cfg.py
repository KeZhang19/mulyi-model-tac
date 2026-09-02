from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from tacmap_sensor.sharpa_tacmap_cfg import SharpaTacmapCfg


REPO_ROOT = Path(__file__).resolve().parent
REVO2_ROOT = REPO_ROOT / "assets" / "revo2_system"
REVO2_USD_ORIGINAL = REVO2_ROOT / "urdf" / "revo2_right.usd"
REVO2_USD_UNINSTANCEABLE = REVO2_ROOT / "urdf" / "revo2_right_uninstanceable.usd"
REVO2_USD = REVO2_USD_ORIGINAL


@configclass
class Revo2TactilePressEnvCfg(DirectRLEnvCfg):
    """Minimal Revo2 tactile press test.

    The object is positioned from the selected tactile map center and pressed
    inward along the average outward tactile normal.
    """

    episode_length_s = 20.0
    decimation = 4
    action_space = 1
    observation_space = 1
    state_space = 0
    seed = 42

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -0.05),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.75, replicate_physics=False)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(REVO2_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=True,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "right_thumb_metacarpal_joint": 0.0,
                "right_thumb_proximal_joint": 0.0,
                "right_thumb_distal_joint": 0.0,
                "right_index_proximal_joint": 0.0,
                "right_index_distal_joint": 0.0,
                "right_middle_proximal_joint": 0.0,
                "right_middle_distal_joint": 0.0,
                "right_ring_proximal_joint": 0.0,
                "right_ring_distal_joint": 0.0,
                "right_pinky_proximal_joint": 0.0,
                "right_pinky_distal_joint": 0.0,
            },
        ),
        actuators={
            "revo2_hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*_joint"],
                effort_limit_sim=2.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    presser_name = "cylinder_D4"
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(REPO_ROOT / "assets" / "presser" / f"{presser_name}.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0001,
                rest_offset=-0.0005,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.55),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    finger: str = "middle"
    touch_link: str = "right_middle_touch_link"
    points_npy: str = str(REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy")
    normals_npy: str = str(REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy")
    object_rot_in_touch_frame = (0.7071067811801017, -3.019609190115596e-06, 0.7071067811800986, -3.0196091876020132e-06)
    object_flip_quat_in_touch_frame = (0.0, 0.0, 1.0, 0.0)
    compliant_contact_stiffness: float = 280.0
    compliant_contact_damping: float = 20.0
    touch_contact_offset: float = 0.0002
    touch_rest_offset: float = -0.0001
    touch_collision_paths = [
        "right_thumb_touch_link/collisions",
        "right_index_touch_link/collisions",
        "right_middle_touch_link/collisions",
        "right_ring_touch_link/collisions",
        "right_pinky_touch_link/collisions",
    ]
    press_start_offset: float = 0.025
    press_end_offset: float = 0.018
    press_steps: int = 240
    press_slide_distance: float = 0.0
    press_slide_steps: int = 0
    press_slide_axis_l: tuple[float, float, float] = (0.0, 1.0, 0.0)

    resolution_step: int = 1
    tacmap_ray_mode: str = "surface_normal"
    tacmap_link_surface_width: int = 240
    tacmap_link_surface_height: int = 240
    tacmap_link_surface_ray_axis: str = "+x"
    tacmap_link_surface_ray_direction: tuple[float, float, float] | None = None
    tacmap_link_surface_grid_u_axis: str = "+y"
    tacmap_link_surface_grid_v_axis: str = "+z"
    tacmap_link_surface_grid_u_size: float = 0.014
    tacmap_link_surface_grid_v_size: float = 0.020
    tacmap_link_surface_grid_center: tuple[float, float, float] = (-0.008, 0.0, 0.0012)
    enable_tactile: bool = True
    enable_deform_vis: bool = True
    enable_deform: bool = True
    enable_touch_materials: bool = True
    enable_contact_sensor: bool = True
    tacmap_use_xform_anchor: bool = False
    tacmap_anchor_prim_path: str = "/World/envs/env_.*/TacmapAnchor"
    tacmap_surface_prim_path: str = ""
    tacmap_surface_track_mesh_transforms: bool = True

    contact_sensor = [
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_touch_link",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=200,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]

    vbts_sensor = [
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_touch_link",
            mesh_prim_paths=[SharpaTacmapCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/object/geometry/mesh")],
            update_period=0.0,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["distance_along_normal", "distance_along_normal_raw"],
            points_npy=str(REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_point.npy"),
            normals_npy=str(REPO_ROOT / "assets" / "tactilesensor_map" / "right_middle_touch_normal.npy"),
            resolution_step=resolution_step,
            max_distance=0.015,
            debug_viz=True,
            debug_viz_max_points=15000,
            debug_viz_color_mode="stripes",
            debug_viz_stripe_interval=10,
            debug_viz_normals=True,
            debug_viz_normal_stride=40,
            correction_scale=1e-3,
        )
    ]

    env_info = {}
